---
marp: true
theme: gaia
class: lead
backgroundColor: #fff
paginate: true
header: "CCTV 기반 시각장애인 안내 시스템"
footer: "코디세이 AI 올인원 텀프로젝트"
style: |
  section { font-family: 'Malgun Gothic', sans-serif; }
  h1 { color: #2c3e50; }
  h2 { color: #34495e; }
  strong { color: #e74c3c; }
---

# CCTV 기반 시각장애인 안내 시스템
## (PC 서버 Pose ID 인식 MVP)

<br>

### 2025. 11. 28.
### 팀 삼무(Samu)

---

## 1. 기획 의도 및 배경

### 🚧 시각장애인의 보행 한계
- **'점'**만 보는 인식: 흰 지팡이, 스마트폰 앱은 바로 앞 장애물만 감지.
- **전체 상황 인지 불가**: 다가오는 자전거, 멀리 있는 공사 현장 등 위험 요소 사각지대.

### 📱 스마트폰 단독 처리의 한계
- 배터리 소모 및 발열 심함.
- 복잡한 Re-ID 연산(재식별) 수행 불가 → **PC 서버 중앙 처리 방식**으로 피벗(Pivot).

---

## 2. 솔루션: CCTV를 '제3의 눈'으로

### 👁️ 능동형 보행 보조 시스템
1. **CCTV 영상 수집**: 기존 인프라 활용 (사각지대 최소화).
2. **PC 서버 분석**: 고성능 GPU로 **Pose ID** 정밀 추적.
3. **음성 안내 전송**: 스마트폰은 '스피커' 역할만 수행 (경량화).

> "사고가 난 뒤에 보는 CCTV가 아니라, **사고를 막아주는 CCTV**"

---

## 3. 핵심 기술: Pose ID Tracking

### 🧩 왜 YOLO만으로는 안 되는가?
- 일반 객체 탐지(YOLO)는 사람이 겹치거나 가려지면 **ID가 수시로 바뀜**.
- 시각장애인에게 "전방 3m"라고 했다가 ID가 바뀌면 엉뚱한 안내를 하게 됨.

### 💡 해결책: Pose Similarity + Dual Layer
- **YOLO ID**: 단기 추적 (Short-term)
- **Pose ID**: 장기 추적 (Long-term, Re-ID)
- **Scale Smoothing**: 원거리/근거리 크기 변화 보정 (30프레임 평균).

---

## 4. 기술 검증 결과 (Performance)

### 📊 안정성 테스트 (Stability Score)
- **실험 환경**: 다수 인원이 교차 보행하는 CCTV 영상 (5분).
- **YOLO 단독**: ID 변경 **164회** (불안정).
- **Pose Tracker**: ID 변경 **15회** (매우 안정).

### 🏆 결과
- **YOLO 대비 약 11배 안정성 확보**.
- 완전 가려짐(Full Occlusion) 후 재등장 시에도 **동일 ID 매칭 성공**.

---

## 5. 시스템 아키텍처

```mermaid
graph LR
    CCTV[CCTV Camera] -->|RTSP| PC[PC Server\n(YOLO+PoseTracker)]
    PC -->|Detection Data| LOG[Log DB]
    PC -->|Voice Guidance| APP[Mobile App]
    APP -->|TTS Output| USER((User))
```

- **Input**: RTSP CCTV Stream
- **Core**: YOLO11-Pose + Custom Tracker Engine
- **Output**: WebSocket Real-time Message

---

## 6. 시연 (Demonstration)

<br>

![bg right:40% fit](https://via.placeholder.com/400x800?text=Tracking+Screen)

### 🎥 주요 관전 포인트
1. **HUD 통계**: YOLO ID가 미친 듯이 바뀔 때 Pose ID는 유지됨.
2. **ID Color**: 사람이 겹쳐도 고유 색상(ID)이 변하지 않음.
3. **Re-ID**: 문 뒤로 사라졌다가 나와도 같은 ID 부여.

---

## 7. 기대 효과

### ✅ 사용자 측면
- **심리적 안정감**: "시스템이 나를 지켜보고 있다"는 신뢰.
- **사각지대 해소**: 지팡이가 닿지 않는 원거리 위험 사전 경고.

### ✅ 인프라 측면
- **저비용 고효율**: 고가의 개별 센서 없이 S/W 업데이트만으로 구현.
- **데이터 활용**: 시각장애인 이동 패턴 분석으로 도시 설계 개선.

---

## 8. 향후 계획 (Roadmap)

### 📅 Sprint 2 (Next Week)
- **FastAPI 서버 구축**: 트래커 엔진을 API 서버로 래핑.
- **WebSocket 연동**: 실시간 데이터 전송 파이프라인 구축.

### 📅 Sprint 3 (Final)
- **Mobile App 프로토타입**: 음성 안내(TTS) 기능 구현.
- **필드 테스트**: 실제 복도 환경에서의 E2E 테스트.

---

# Q & A
## 경청해 주셔서 감사합니다.

### 팀 삼무 (Samu)







