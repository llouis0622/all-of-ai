import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import { katex } from '@mdit/plugin-katex'

export default withMermaid(defineConfig({
  title: 'All of AI',
  description: '인공지능 전 영역을 체계적으로 정리한 지식베이스',
  base: '/all-of-ai/',

  head: [
    [
      'link',
      {
        rel: 'stylesheet',
        href: 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css',
      },
    ],
  ],

  markdown: {
    config: (md) => {
      md.use(katex)
    },
  },

  mermaid: {},

  themeConfig: {
    search: {
      provider: 'local',
    },

    nav: [
      { text: '홈', link: '/' },
    ],

    sidebar: [
      {
        text: 'I. 인공지능 기초',
        items: [
          {
            text: 'Chapter 01. 인공지능과 학습',
            collapsed: false,
            items: [
              { text: 'Lecture 01. 인공지능이란 무엇인가', link: '/ch01/lecture01' },
              { text: 'Lecture 02. 학습의 개념', link: '/ch01/lecture02' },
              { text: 'Lecture 03. 학습 패러다임', link: '/ch01/lecture03' },
            ],
          },
          {
            text: 'Chapter 02. 일반화와 이론적 기초',
            collapsed: false,
            items: [
              { text: 'Lecture 04. 일반화 문제', link: '/ch02/lecture04' },
              { text: 'Lecture 05. 편향과 분산', link: '/ch02/lecture05' },
            ],
          },
        ],
      },
      {
        text: 'II. 머신러닝과 딥러닝 핵심',
        items: [
          {
            text: 'Chapter 03. 머신러닝의 기본 구조',
            collapsed: false,
            items: [
              { text: 'Lecture 06. 경험적 위험 최소화', link: '/ch03/lecture06' },
              { text: 'Lecture 07. 전통적 머신러닝 모델', link: '/ch03/lecture07' },
            ],
          },
          {
            text: 'Chapter 04. 딥러닝과 표현 학습',
            collapsed: false,
            items: [
              { text: 'Lecture 08. 신경망의 원리', link: '/ch04/lecture08' },
              { text: 'Lecture 09. 표현 학습', link: '/ch04/lecture09' },
              { text: 'Lecture 10. 주요 신경망 구조', link: '/ch04/lecture10' },
            ],
          },
        ],
      },
      {
        text: 'III. 생성 모델과 파운데이션 모델',
        items: [
          {
            text: 'Chapter 05. 생성 모델',
            collapsed: false,
            items: [
              { text: 'Lecture 11. 생성이란 무엇인가', link: '/ch05/lecture11' },
              { text: 'Lecture 12. 생성 모델의 주요 접근법', link: '/ch05/lecture12' },
            ],
          },
          {
            text: 'Chapter 06. 대규모 언어 모델',
            collapsed: false,
            items: [
              { text: 'Lecture 13. LLM의 본질', link: '/ch06/lecture13' },
              { text: 'Lecture 14. 스케일링과 능력', link: '/ch06/lecture14' },
              { text: 'Lecture 15. 정렬과 미세조정', link: '/ch06/lecture15' },
            ],
          },
        ],
      },
      {
        text: 'IV. 문제 영역과 의사결정',
        items: [
          {
            text: 'Chapter 07. 문제 영역별 인공지능',
            collapsed: false,
            items: [
              { text: 'Lecture 16. 비전/언어/오디오 문제의 구조', link: '/ch07/lecture16' },
            ],
          },
          {
            text: 'Chapter 08. 강화학습',
            collapsed: false,
            items: [
              { text: 'Lecture 17. 강화학습의 기본 개념', link: '/ch08/lecture17' },
              { text: 'Lecture 18. 강화학습의 주요 방법론', link: '/ch08/lecture18' },
            ],
          },
        ],
      },
      {
        text: 'V. AI 시스템과 에이전트',
        items: [
          {
            text: 'Chapter 09. LLM 시스템',
            collapsed: false,
            items: [
              { text: 'Lecture 19. LLM 기반 시스템 설계', link: '/ch09/lecture19' },
            ],
          },
          {
            text: 'Chapter 10. 에이전트형 인공지능',
            collapsed: false,
            items: [
              { text: 'Lecture 20. 에이전트형 인공지능', link: '/ch10/lecture20' },
            ],
          },
        ],
      },
      {
        text: 'VI. 인공지능 수학과 컴퓨터 과학',
        items: [
          {
            text: 'Chapter 11. 인공지능 수학',
            collapsed: false,
            items: [
              { text: 'Lecture 21. 선형대수와 표현', link: '/ch11/lecture21' },
              { text: 'Lecture 22. 확률적 모델링', link: '/ch11/lecture22' },
              { text: 'Lecture 23. 최적화와 학습', link: '/ch11/lecture23' },
            ],
          },
          {
            text: 'Chapter 12. 컴퓨터 과학 기초',
            collapsed: false,
            items: [
              { text: 'Lecture 24. 자료구조', link: '/ch12/lecture24' },
              { text: 'Lecture 25. 컴퓨터 구조', link: '/ch12/lecture25' },
              { text: 'Lecture 26. 운영체제', link: '/ch12/lecture26' },
              { text: 'Lecture 27. 네트워크', link: '/ch12/lecture27' },
              { text: 'Lecture 28. 데이터베이스', link: '/ch12/lecture28' },
            ],
          },
        ],
      },
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/llouis0622/all-of-ai' },
    ],
  },
}))
