# 협업 개발 환경 Setting Guideline

- **프로젝트 기간동안 VSCode를 사용할 예정이니 PC에 없다면 설치해주시길 바랍니다.**
- 설치 링크: https://code.visualstudio.com/

---------------------------------------------------------

## macOS

- **터미널 실행 후 아래 명령어 한 줄씩 입력**(복붙 가능)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install uv
brew install git
git clone https://github.com/ryu-rxxt/SKY_IE_Project_2025.git
cd SKY_IE_Project_2025
uv venv --python 3.11.9
source .venv/bin/activate
uv sync --python .venv/bin/python
```

## Windows OS

- https://git-scm.com 에서 Git 다운로드 및 설치(이미 있으신 분은 생략하셔도 됩니다)

- **Git Bash 실행 후 아래 명령어 한 줄씩 입력**(복붙 가능)
```bash
irm https://astral.sh/uv/install.ps1 | iex
git clone https://github.com/ryu-rxxt/SKY_IE_Project_2025.git
cd SKY_IE_Project_2025
uv venv --python 3.11.9
source .venv/bin/activate
uv sync --python .venv/Scripts/python.exe
```
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

- VS Code 실행
- `ctrl` + `(숫자 1 좌측에 위치) 누르고 화면 하단에 터미널 창 생기는지 확인
- 터미널에 아래 **명령어를 한 줄씩 입력**하면 `main.py`가 실행됨
```
cd ~/SKY_IE_Project_2025
code .
```
- `code .` 명령어가 안되면 `Command Palette`(단축키: `Ctrl` + `Shift` + `P` 또는 `F1` -> `Shell Command: Install ‘code’ command in PATH` 클릭 후 재시도

---------------------------------------------------------

## FAQ

Q. uv가 안된다고 나와요.
- `uv --version`을 입력해서 숫자가 안 나오는 경우 설치 스크립트를 다시 실행하시면 됩니다.

Q. 가상환경이 안 켜져요.
- macOS는 터미널에 `source .venv/bin/activate` 입력
- Windows OS는 `Git Bash`에 `. .venv/Scripts/Activate.ps1` 입력 (`.`도 꼭 포함해주세요)

Q. Python은 설치 안 해도 돼요?
- 네. `uv`가 자동으로 3.11.9 버전을 다운로드하여 가상환경에 설치해줍니다.

---------------------------------------------------------

# 작업 전후 Guideline
- 작업 시작 전과 마무리 후에는 **Project 파일이 반드시 GitHub와 동기화** 되어있어야 합니다.
- 간단하여 **오래걸리지 않으니 잊지 않고 반드시** 아래 적어둔 명령어를 입력해주시길 바랍니다.
- 위와 마찬가지로 Windows의 경우 `Git Bash`, Mac의 경우 `터미널`에 입력해주시면 됩니다.

---------------------------------------------------------

## 작업 시작 전
```
git pull origin main
uv sync
```
---------------------------------------------------------
## 새로운 패키지(모듈)를 코드에 추가한 경우
- 예를 들어 코드에 `import numpy`, `import tensorflow`가 없었는데 내가 추가한 경우
```
uv add numpy tensorflow
git add pyproject.toml uv.lock
git commit -m "package add"
git push origin (각자 이름 이니셜 ex: kh)
```
---------------------------------------------------------
## 작업 마무리 후
```
git add .
# 기능 추가한 경우
git commit -m "Add: ~~ 기능 추가"
# 오류 수정한 경우
git commit -m "Fix: ~~ 오류 수정"
git push origin (각자 이름 이니셜 ex: kh)
```
---------------------------------------------------------
# References
- uv 공식 문서: https://astral.sh/blog/uv/
- Python 공식 사이트: https://www.python.org/
