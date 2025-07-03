# 협업 개발 환경 Setting Guideline

- **프로젝트 기간동안 VSCode를 사용할 예정이니 PC에 없다면 설치해주시길 바랍니다**

---------------------------------------------------------

## macOS
- 터미널 실행 후 아래 명령어 한 줄씩 입력(복붙 가능)
  
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install uv
brew install git
git clone https://github.com/ryu-rxxt/SKY_IE_Project_2025.git
cd SKY_IE_Project_2025
uv venv --python 3.11.9
source .venv/bin/activate
uv sync --python .venv/bin/python

## Windows OS
- Git 설치 (이미 있다면 건너뛰기)
-> https://git-scm.com 에서 다운로드 및 설치

- Git Bash 실행 후 아래 명령어 한 줄씩 입력(복붙 가능)
irm https://astral.sh/uv/install.ps1 | iex
git clone https://github.com/ryu-rxxt/SKY_IE_Project_2025.git
cd SKY_IE_Project_2025
uv venv --python 3.11.9
source .venv/bin/activate
uv sync --python .venv/Scripts/python.exe

---------------------------------------------------------

## 폴더 구조 및 파일 설명

```text
SKY_IE_Project_2025/
├── data/                     ← 제공받은 데이터 파일
├── pyproject.toml            ← 의존성 관리
├── requirements.txt          ← 의존성 관리
├── .gitignore                ← 추적 제외 파일 목록
├── README.md                 ← 지금 읽고 있는 파일
├── .vscode/settings.json     ← VS Code 설정
├── src/
│   └── sky_ie_project/
│       └── main.py           ← 실행할 메인 코드
```

---------------------------------------------------------

## 프로젝트 실행 방법

- 가상환경이 활성화된 상태에서 아래 명령어 입력:
python src/sky_ie_project/main.py

 또는

- VS Code에서 `main.py` 열고 우측 상단 ▶ 버튼 클릭

---------------------------------------------------------

## FAQ

Q. uv가 안된다고 나와요.
- `uv --version`을 입력해보세요. 안 나오면 설치가 안 된 거예요. 설치 스크립트를 다시 실행하세요.

Q. 가상환경이 안 켜져요.
- macOS는 터미널에 `source .venv/bin/activate` 입력
- Windows OS는 PowerShell에 `. .venv/Scripts/Activate.ps1` 입력 (`.`도 꼭 포함해주세요)

Q. Python은 설치 안 해도 돼요?
- 네. `uv`가 자동으로 3.11.9 버전을 다운로드하여 가상환경에 설치해줍니다.

---------------------------------------------------------

#
git pull origin main
uv sync

## References
- uv 공식 문서: https://astral.sh/blog/uv/
- Python 공식 사이트: https://www.python.org/
