"""한국어 피싱 문자 분류 모델을 학습하고 joblib 파일로 저장한다."""

import argparse
import csv
import io
import logging
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from time import perf_counter
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
logger = logging.getLogger(__name__)


def read_messages(dataset, input_path):
    url, label_map = DATASETS[dataset]
    csv_name = f"{dataset}.csv"
    if input_path is None:
        local_csv = Path(csv_name)
        if local_csv.is_file():
            input_path = local_csv

    if input_path is not None:
        logger.info("[DATA] 데이터 읽는 중")
        logger.debug("[DATA] path=%s", input_path.resolve())
        data = input_path.read_bytes()
    else:
        logger.info("[DATA] 데이터 다운로드 중")
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
    skipped = 0
    for row in reader:
        text = row["content"]
        source_label = (row["class"] or "").strip()
        if text and text.strip() and source_label in label_map:
            rows.append((text, label_map[source_label]))
        else:
            skipped += 1

    counts = Counter(label for _, label in rows)
    logger.debug(
        "[DATA] loaded dataset=%s samples=%d normal=%d phishing=%d skipped=%d",
        dataset, len(rows), counts[0], counts[1], skipped,
    )
    return rows


def split_messages(rows):
    logger.info("[SPLIT] 데이터 정리 및 분리 중")
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
    group_count = len(set(groups))
    if group_count < 5 or min(label_counts[0], label_counts[1]) < 5:
        raise ValueError("5개 그룹으로 나누려면 정상·피싱 예문과 서로 다른 문구가 충분히 필요합니다.")

    texts = np.asarray(texts, dtype=object)
    labels = np.asarray(labels)
    logger.debug("[SPLIT] start samples=%d removed=%d groups=%d", len(texts), len(rows) - len(texts), group_count)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = [held_out for _, held_out in splitter.split(texts, labels, groups)]
    test, validation = folds[:2]
    train = np.concatenate(folds[2:])

    for name, indices in [("train", train), ("validation", validation), ("test", test)]:
        counts = Counter(labels[indices].tolist())
        if set(counts) != {0, 1}:
            raise ValueError(f"{name} 구간에 정상과 피싱 라벨이 모두 필요합니다.")
        logger.debug("[SPLIT] set=%s samples=%d normal=%d phishing=%d", name, len(indices), counts[0], counts[1])

    return texts, labels, train, validation, test


def train_and_evaluate(texts, labels, train, validation, test):
    candidates = {
        "CountVectorizer+MultinomialNB": make_pipeline(CountVectorizer(), MultinomialNB()),
        "TfidfVectorizer+ComplementNB": make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
            ComplementNB(alpha=0.5),
        ),
    }

    validation_scores = {}
    for name, candidate in candidates.items():
        logger.info("[TRAIN] %s 학습 중", type(candidate[-1]).__name__)
        started = perf_counter()
        candidate.fit(texts[train], labels[train])
        scores = candidate.predict_proba(texts[validation])[:, 1]
        validation_scores[name] = average_precision_score(labels[validation], scores)
        feature_count = len(candidate[0].vocabulary_)
        logger.debug(
            "[TRAIN] done model=%s validation_ap=%.3f features=%d elapsed=%.1fs",
            name, validation_scores[name], feature_count, perf_counter() - started,
        )

    selected_name = max(validation_scores, key=validation_scores.get)
    logger.debug("[TRAIN] selected model=%s validation_ap=%.3f", selected_name, validation_scores[selected_name])

    logger.info("[CALIBRATE] 예측 확률 보정 중")
    logger.debug("[CALIBRATE] method=sigmoid samples=%d", len(validation))
    # 학습된 분류기를 고정하고 검증 데이터로만 확률을 보정한다.
    model = CalibratedClassifierCV(FrozenEstimator(candidates[selected_name]), method="sigmoid")
    model.fit(texts[validation], labels[validation])

    logger.info("[EVAL] 모델 성능 평가 중")
    scores = model.predict_proba(texts[test])[:, 1]
    predictions = model.predict(texts[test])
    report = classification_report(
        labels[test], predictions, labels=[0, 1], target_names=["normal", "phishing"],
        output_dict=True, zero_division=0,
    )
    logger.debug("[EVAL] ap=%.3f accuracy=%.3f", average_precision_score(labels[test], scores), report["accuracy"])
    for label in ("normal", "phishing"):
        metrics = report[label]
        logger.debug(
            "[EVAL] label=%s precision=%.3f recall=%.3f f1=%.3f support=%d",
            label, metrics["precision"], metrics["recall"], metrics["f1-score"], metrics["support"],
        )
    true_normal, false_alarm, missed, detected = confusion_matrix(labels[test], predictions, labels=[0, 1]).ravel()
    logger.debug("[EVAL] tn=%d fp=%d fn=%d tp=%d", true_normal, false_alarm, missed, detected)
    return model


def save_model(model, output_path):
    logger.info("[EXPORT] 모델 저장 중")
    examples = [
        "안녕하세요! 내일 학교 끝나고 도서관에서 만날까요?",
        "[광고] 오늘만 전 상품 할인! 무료 쿠폰을 신청하세요.",
        "[웹발신] 배송 주소 오류로 배송이 지연되었습니다. 주소를 확인해 주세요.",
    ]
    probabilities = model.predict_proba(examples)
    joblib.dump(model, output_path, compress=3)
    reloaded = joblib.load(output_path)
    np.testing.assert_allclose(reloaded.predict_proba(examples), probabilities)
    logger.info("[EXPORT] 저장 완료 · %s", output_path)
    logger.debug(
        "[EXPORT] saved path=%s verified_samples=%d scikit-learn=%s joblib=%s",
        output_path.resolve(), len(examples), sklearn.__version__, joblib.__version__,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default=DEFAULT_DATASET, help="데이터셋과 라벨 체계 선택")
    parser.add_argument("--input", type=Path, help="로컬 학습용 CSV 또는 ZIP 경로")
    parser.add_argument("--output", type=Path, default=Path("spam_classifier.joblib"), help="모델을 저장할 경로")
    args = parser.parse_args()

    if args.output.resolve() == (args.input or Path(f"{args.dataset}.csv")).resolve():
        parser.error("학습 데이터와 모델 저장 경로는 달라야 합니다.")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout,
    )
    try:
        rows = read_messages(args.dataset, args.input)
        model = train_and_evaluate(*split_messages(rows))
        save_model(model, args.output)
    except (OSError, ValueError, csv.Error, BadZipFile) as error:
        parser.exit(1, f"실행 오류: {error}\n")


if __name__ == "__main__":
    main()
