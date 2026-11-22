"""한국어 피싱 문자 분류 모델을 학습하고 joblib 파일로 저장한다."""

import argparse
import csv
import io
import re
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

import joblib
import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import make_pipeline


DATASETS = {
    "KOR_phishing_data_raw": (
        "https://raw.githubusercontent.com/Ez-Sy01/KOR_phishing_Detect-Dataset/main/data/KOR_phishing_data_raw.zip",
        {"0": 0, "1": 1},
    ),
    "lgaidataset": (
        "https://huggingface.co/datasets/meal-bbang/Korean_message/resolve/main/lgaidataset.csv",
        {"1": 0, "2": 1},
    ),
}
DEFAULT_DATASET = "KOR_phishing_data_raw"


def read_messages(dataset, input_path):
    url, label_map = DATASETS[dataset]
    csv_name = f"{dataset}.csv"
    if input_path is None:
        local_csv = Path(csv_name)
        if local_csv.is_file():
            input_path = local_csv

    if input_path is not None:
        data = input_path.read_bytes()
    else:
        print(f"{csv_name} 다운로드 중…", flush=True)
        try:
            with urlopen(url, timeout=25) as response:
                data = response.read()
        except (URLError, TimeoutError) as error:
            raise ValueError(f"데이터를 받지 못했습니다. --input으로 CSV나 ZIP 경로를 지정하세요. ({error})") from error

    if data.startswith(b"PK\x03\x04"):
        with ZipFile(io.BytesIO(data)) as archive:
            matches = [entry for entry in archive.namelist() if Path(entry).name == csv_name]
            if len(matches) != 1:
                raise ValueError(f"ZIP 안에 {csv_name} 파일이 하나 있어야 합니다.")
            data = archive.read(matches[0])

    try:
        csv_text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        csv_text = data.decode("cp949")

    reader = csv.DictReader(io.StringIO(csv_text, newline=""))
    if not {"content", "class"}.issubset(reader.fieldnames or []):
        raise ValueError("CSV에 content(본문)와 class(라벨) 열이 있어야 합니다.")

    rows = []
    for row in reader:
        text = row["content"]
        source_label = (row["class"] or "").strip()
        if text and text.strip() and source_label in label_map:
            rows.append((text, label_map[source_label]))

    counts = Counter(label for _, label in rows)
    print(f"{dataset}: 사용 가능한 원본 {len(rows):,}건")
    print(f"라벨 분포: 정상 {counts[0]:,}건, 피싱 {counts[1]:,}건")
    return rows


def split_messages(rows):
    # 공백이나 글자 표기만 다른 본문은 하나로 보고, 라벨이 서로 다른 중복은 제외한다.
    unique = {}
    for text, label in rows:
        key = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
        if key not in unique:
            unique[key] = (text, label)
        elif unique[key] is not None and unique[key][1] != label:
            unique[key] = None

    texts, labels, groups = [], [], []
    for key, example in unique.items():
        if example is None:
            continue
        text, label = example
        # 링크나 숫자만 다른 문자가 학습과 평가 양쪽에 들어가는 것을 막는다.
        template = re.sub(r"(?:https?://|www\.)\S+", "<url>", key)
        groups.append(re.sub(r"\d+", "0", template))
        texts.append(text)
        labels.append(label)

    label_counts = Counter(labels)
    if len(set(groups)) < 5 or min(label_counts[0], label_counts[1]) < 5:
        raise ValueError("5개 그룹으로 나누려면 정상·피싱 예문과 서로 다른 문구가 충분히 필요합니다.")

    texts = np.asarray(texts, dtype=object)
    labels = np.asarray(labels)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = [held_out for _, held_out in splitter.split(texts, labels, groups)]
    test, validation = folds[:2]
    train = np.concatenate(folds[2:])

    for name, indices in [("학습", train), ("검증·확률 보정", validation), ("최종 평가", test)]:
        counts = Counter(labels[indices].tolist())
        if set(counts) != {0, 1}:
            raise ValueError(f"{name} 구간에 정상과 피싱 라벨이 모두 필요합니다.")
        print(f"{name}: {len(indices):,}건 (정상 {counts[0]:,}, 피싱 {counts[1]:,})")

    return texts, labels, train, validation, test


def train_and_evaluate(texts, labels, train, validation, test):
    candidates = {
        "단어 빈도 + MultinomialNB": make_pipeline(CountVectorizer(), MultinomialNB()),
        "글자 묶음 TF-IDF + ComplementNB": make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
            ComplementNB(alpha=0.5),
        ),
    }

    validation_scores = {}
    for name, candidate in candidates.items():
        candidate.fit(texts[train], labels[train])
        scores = candidate.predict_proba(texts[validation])[:, 1]
        validation_scores[name] = average_precision_score(labels[validation], scores)
        print(f"검증 AP | {name} | {validation_scores[name]:.3f}")

    selected_name = max(validation_scores, key=validation_scores.get)
    # 학습된 분류기를 고정하고 검증 데이터로만 확률을 보정한다.
    model = CalibratedClassifierCV(FrozenEstimator(candidates[selected_name]), method="sigmoid")
    model.fit(texts[validation], labels[validation])

    scores = model.predict_proba(texts[test])[:, 1]
    predictions = model.predict(texts[test])
    print(f"\n선택한 모델: {selected_name}")
    print(f"최종 평가 AP: {average_precision_score(labels[test], scores):.3f}")
    print(classification_report(
        labels[test], predictions, labels=[0, 1], target_names=["정상", "피싱"], digits=3, zero_division=0,
    ))
    print("혼동 행렬(행은 실제, 열은 예측, 순서는 정상·피싱)")
    print(confusion_matrix(labels[test], predictions, labels=[0, 1]))
    return model


def save_model(model, output_path):
    examples = [
        "안녕하세요! 내일 학교 끝나고 도서관에서 만날까요?",
        "[광고] 오늘만 전 상품 할인! 무료 쿠폰을 신청하세요.",
        "[웹발신] 배송 주소 오류로 배송이 지연되었습니다. 주소를 확인해 주세요.",
    ]
    probabilities = model.predict_proba(examples)
    for text, probability in zip(examples, probabilities[:, 1]):
        print(f"> {text}\n[ 스팸 확률 : {probability:.1%} ]\n")

    joblib.dump(model, output_path, compress=3)
    reloaded = joblib.load(output_path)
    np.testing.assert_allclose(reloaded.predict_proba(examples), probabilities)
    print(f"저장 완료: {output_path.resolve()} (scikit-learn {sklearn.__version__}, joblib {joblib.__version__})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default=DEFAULT_DATASET, help="데이터셋과 라벨 체계 선택")
    parser.add_argument("--input", type=Path, help="로컬 학습용 CSV 또는 ZIP 경로")
    parser.add_argument("--output", type=Path, default=Path("spam_classifier.joblib"), help="모델을 저장할 경로")
    args = parser.parse_args()

    if args.output.resolve() == (args.input or Path(f"{args.dataset}.csv")).resolve():
        parser.error("학습 데이터와 모델 저장 경로는 달라야 합니다.")

    try:
        rows = read_messages(args.dataset, args.input)
        model = train_and_evaluate(*split_messages(rows))
        save_model(model, args.output)
    except (OSError, ValueError, csv.Error, BadZipFile) as error:
        parser.exit(1, f"실행 오류: {error}\n")


if __name__ == "__main__":
    main()
