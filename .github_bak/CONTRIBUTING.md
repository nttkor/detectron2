# 기여 가이드

detectron2 주차 공간 탐지 프로젝트에 기여해주셔서 감사합니다!

## 개발 환경 설정

### 필수 요구사항
- Python 3.10 이상
- PyTorch
- Detectron2
- CUDA (GPU 사용 시)

### 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/your-username/detectron2.git
cd detectron2
```

2. 가상 환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 의존성 설치
```bash
pip install -r requirements.txt
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## 개발 워크플로우

1. 새로운 브랜치 생성
```bash
git checkout -b feature/your-feature-name
```

2. 변경사항 커밋
```bash
git add .
git commit -m "feat: your feature description"
```

3. 브랜치 푸시 및 PR 생성
```bash
git push origin feature/your-feature-name
```

## 코드 스타일

- Python 코드는 PEP 8 스타일 가이드를 따릅니다
- 코드 포맷팅은 Black을 사용합니다
- import 정렬은 isort를 사용합니다

## 커밋 메시지 규칙

커밋 메시지는 다음 형식을 따릅니다:
```
<type>: <subject>

<body>
```

타입:
- `feat`: 새로운 기능
- `fix`: 버그 수정
- `docs`: 문서 변경
- `style`: 코드 포맷팅
- `refactor`: 코드 리팩토링
- `test`: 테스트 추가/수정
- `chore`: 빌드 프로세스 또는 보조 도구 변경

## Pull Request 가이드라인

1. PR 제목은 명확하고 간결하게 작성해주세요
2. 변경사항을 자세히 설명해주세요
3. 관련 이슈가 있다면 링크해주세요
4. 테스트를 추가하거나 기존 테스트를 통과하는지 확인해주세요

## 질문이 있으신가요?

이슈를 생성하거나 프로젝트 관리자에게 문의해주세요.

