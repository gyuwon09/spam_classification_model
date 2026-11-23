"""저장한 모델로 입력한 문자의 스팸 확률을 확인한다."""

import argparse
import pickle
import sys
from pathlib import Path


def find_spam_column(classes, requested_label):
    label_names = [str(label) for label in classes]
    labels = [name.casefold() for name in label_names]
    label_set = set(labels)

    # 데이터에 따라 정상·스팸 라벨이 0·1 또는 1·2로 저장된다.
    if requested_label is not None:
        spam_labels = {requested_label.casefold()}
    elif label_set == {"0", "1"}:
        spam_labels = {"1"}
    elif label_set == {"1", "2"}:
        spam_labels = {"2"}
    else:
        spam_labels = {"spam", "smishing", "phishing", "스팸", "스미싱", "피싱"}

    matches = [column for column, label in enumerate(labels) if label in spam_labels]
    if len(matches) != 1:
        available = ", ".join(label_names)
        raise ValueError(f"--spam-label로 스팸 라벨을 지정하세요. 모델의 라벨은 {available}입니다.")
    return matches[0]


def show_spam_probability(model, message, spam_column):
    probabilities = model.predict_proba([message])[0]
    print(f"[ 스팸 확률 : {probabilities[spam_column]:.1%} ]\n")


def main():
    default_model = Path(__file__).resolve().with_name("model.joblib")
    parser = argparse.ArgumentParser(description="저장한 스팸 분류 모델을 테스트합니다.")
    parser.add_argument(
        "--model",
        type=Path,
        default=default_model,
        help=f"모델 파일 경로 (기본값: 이 프로그램과 같은 폴더의 {default_model.name})",
    )
    parser.add_argument("--text", help="메시지 한 건을 바로 테스트합니다.")
    parser.add_argument("--spam-label", help="스팸으로 간주할 모델 라벨을 지정합니다.")
    args = parser.parse_args()

    if args.text is not None and not args.text.strip():
        parser.error("--text에 비어 있지 않은 메시지를 입력하세요.")
    if not args.model.is_file():
        parser.error(f"모델 파일을 찾을 수 없습니다. --model로 경로를 지정하세요. ({args.model})")

    try:
        import joblib
    except ModuleNotFoundError:
        parser.exit(1, "joblib이 필요합니다. python -m pip install scikit-learn joblib\n")

    try:
        # 직접 저장했거나 출처를 신뢰할 수 있는 모델 파일만 연다.
        model = joblib.load(args.model)
    except (OSError, ImportError, AttributeError, ValueError, EOFError, pickle.UnpicklingError) as error:
        print(
            f"모델을 불러오지 못했습니다. 파일과 학습 환경의 패키지 버전을 확인하세요. ({error})",
            file=sys.stderr,
        )
        return 1

    classes = getattr(model, "classes_", None)
    if classes is None or not callable(getattr(model, "predict_proba", None)):
        parser.error("확률을 예측할 수 있는 학습된 모델을 사용하세요.")
    try:
        spam_column = find_spam_column(classes, args.spam_label)
    except ValueError as error:
        parser.error(str(error))

    try:
        if args.text is not None:
            show_spam_probability(model, args.text, spam_column)
            return 0

        while True:
            message = input("> ")
            if not message.strip():
                return 0
            show_spam_probability(model, message, spam_column)
            print()
    except (EOFError, KeyboardInterrupt):
        print()
        return 0
    except (AttributeError, ValueError, TypeError) as error:
        print(
            f"예측에 실패했습니다. 벡터라이저와 분류기를 함께 저장했는지 확인하세요. ({error})",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
