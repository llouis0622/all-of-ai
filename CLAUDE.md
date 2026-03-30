# CLAUDE.md — All of AI 프로젝트 작업 문서

## 프로젝트 개요

- **프로젝트명**: All of AI
- **목적**: 인공지능 전 영역을 체계적으로 정리한 지식베이스
- **배포 URL**: https://llouis0622.github.io/all-of-ai/
- **저장소**: https://github.com/llouis0622/all-of-ai

---

## 기술 스택

| 항목 | 내용 |
|---|---|
| 정적 사이트 | VitePress ^1.6.3 |
| 수식 렌더링 | KaTeX (`@mdit/plugin-katex` + `katex`) |
| 다이어그램 | Mermaid (`vitepress-plugin-mermaid`) |
| 검색 | VitePress 로컬 서치 (built-in) |
| 배포 | GitHub Actions → GitHub Pages |

---

## 완료된 작업 (2026-03-28)

### 프로젝트 초기 세팅
- [x] `package.json` — `dev`, `build`, `preview` 스크립트 포함
- [x] `.gitignore` — `node_modules/`, `dist/`, `.vitepress/cache/` 제외
- [x] `README.md` — 로컬 실행 방법 작성
- [x] `CLAUDE.md` — 작업 현황 문서 (이 파일)

### VitePress 설정
- [x] `docs/.vitepress/config.mts` — 사이드바, KaTeX, Mermaid, 로컬 서치, base URL 설정
- [x] `docs/index.md` — 홈페이지 (hero 섹션 + 전체 목차 링크)

### 콘텐츠 구조 (템플릿)
28개 Lecture .md 파일 생성 (모두 빈 템플릿 상태):
- [x] ch01: Lecture 01~03
- [x] ch02: Lecture 04~05
- [x] ch03: Lecture 06~07
- [x] ch04: Lecture 08~10
- [x] ch05: Lecture 11~12
- [x] ch06: Lecture 13~15
- [x] ch07: Lecture 16
- [x] ch08: Lecture 17~18
- [x] ch09: Lecture 19
- [x] ch10: Lecture 20
- [x] ch11: Lecture 21~23
- [x] ch12: Lecture 24~28

### 이미지 디렉토리
- [x] `docs/public/images/ch01/` ~ `ch12/` 생성 (`.gitkeep` 포함)

### 배포 파이프라인
- [x] `.github/workflows/deploy.yml` — `main` 브랜치 push 시 GitHub Pages 자동 배포

### 빌드 검증
- [x] `npm run build` 성공 확인
- [x] `npm run dev` 로컬 실행 확인 (http://localhost:5173/all-of-ai/)

---

## 앞으로 할 것들

### 콘텐츠 작성 (우선순위 순)

각 Lecture .md 파일에 아래 섹션을 채워야 함:
- `## 개요` — 핵심 질문과 학습 목표
- `## 핵심 개념` — 주요 개념 정의 및 설명
- `## 수식` — KaTeX 수식 블록
- `## 시각화` — Mermaid 다이어그램 또는 이미지
- `## 직관적 이해` — 수식 없이 개념의 본질 설명
- `## 참고` — 논문, 자료 링크

#### I. 인공지능 기초
- [x] Lecture 01. 인공지능이란 무엇인가
- [ ] Lecture 02. 학습의 개념
- [ ] Lecture 03. 학습 패러다임
- [ ] Lecture 04. 일반화 문제
- [ ] Lecture 05. 편향과 분산

#### II. 머신러닝과 딥러닝 핵심
- [ ] Lecture 06. 경험적 위험 최소화
- [ ] Lecture 07. 전통적 머신러닝 모델
- [ ] Lecture 08. 신경망의 원리
- [ ] Lecture 09. 표현 학습
- [ ] Lecture 10. 주요 신경망 구조

#### III. 생성 모델과 파운데이션 모델
- [ ] Lecture 11. 생성이란 무엇인가
- [ ] Lecture 12. 생성 모델의 주요 접근법
- [ ] Lecture 13. LLM의 본질
- [ ] Lecture 14. 스케일링과 능력
- [ ] Lecture 15. 정렬과 미세조정

#### IV. 문제 영역과 의사결정
- [ ] Lecture 16. 비전/언어/오디오 문제의 구조
- [ ] Lecture 17. 강화학습의 기본 개념
- [ ] Lecture 18. 강화학습의 주요 방법론

#### V. AI 시스템과 에이전트
- [ ] Lecture 19. LLM 기반 시스템 설계
- [ ] Lecture 20. 에이전트형 인공지능

#### VI. 인공지능 수학과 컴퓨터 과학
- [ ] Lecture 21. 선형대수와 표현
- [ ] Lecture 22. 확률적 모델링
- [ ] Lecture 23. 최적화와 학습
- [ ] Lecture 24. 자료구조
- [ ] Lecture 25. 컴퓨터 구조
- [ ] Lecture 26. 운영체제
- [ ] Lecture 27. 네트워크
- [ ] Lecture 28. 데이터베이스

### 디자인 / UX 개선
- [ ] 커스텀 테마 또는 컬러 설정 (`.vitepress/theme/`)
- [ ] 챕터별 인트로 페이지 (`ch01/index.md` 등) 추가
- [ ] 파트별 인트로 페이지 추가

### 인프라
- [ ] GitHub Pages 소스를 GitHub Actions으로 설정 (레포 Settings → Pages)
- [ ] 첫 배포 후 URL 동작 확인

---

## 로컬 실행

```bash
npm install       # 최초 1회
npm run dev       # 개발 서버 (http://localhost:5173/all-of-ai/)
npm run build     # 정적 빌드
npm run preview   # 빌드 결과 미리보기
```

---

## 디렉토리 구조 요약

```
all-of-ai/
├── .github/workflows/deploy.yml
├── .gitignore
├── CLAUDE.md
├── README.md
├── package.json
└── docs/
    ├── .vitepress/config.mts
    ├── index.md
    ├── ch01/ ~ ch12/          # 28개 Lecture .md
    └── public/images/ch01~12/ # 챕터별 이미지
```
