---
layout: post
title: Mantaflow 설치시 오류들
comments: true
tags : [mantaflow, graphics]
---

## Mantaflow

[http://mantaflow.com/](http://mantaflow.com/)  
유체 시뮬레이션 프레임워크  
> 설치 시 읽어보면 좋을 [가이드](http://www.programmersought.com/article/4597155267/)  
  
---  
---
  
## 해결한 오류들

> Windows10, 64bit 기준

### 1. cmake 빌드 시

가급적  python, numpy, qt include 및 library dir 설정은 수동으로 하는게 편함. (CMakeLists.txt 편집)  
  
그리고 CMakeLists.txt의 PYTHON_LIBRARY dir 있는 부분에 PYTHON_LIBRARIES 로 되있음.

```
list(APPEND F_LIBS ${PYTHON_LIBRARY})
```
  
#### 1) PYTHON PATH

아나콘다 가상환경의 python path 로 하면, 나중에 빌드 후 실행할때 잡히는 건 이상하게 가상환경껄로 안잡히더라...?  

애초에 처음부터 가상환경 아닌 그냥 로컬(?)에 설치된 python 으로 path 잡아야함.
(뒤에도 나옴)
  
#### 2) 32비트, 64비트 확인

QT 및 cmkae 빌더 설정할때도 비트 설정 잘 맞춰야 나중에 빌드 시 꼬이지 않음.  
  
---  

### 2. python37_d.lib(python37_d.dll) 파일 찾을 수 없습니다

솔루션의 manta 프로젝트 파일의 외부 종속성에서,  
pyconfig.h 들어가서 해당부분 주석처리 하기

```
if defined(_DEBUG)  
//pragma comment(lib,"python37_d.lib")
```

---  

### 3. .obj 파일 외부 기호 참조 오류

```
static void __cdecl Manta::PbClass::renameObjects(void)
public: struct _object * __cdecl Pb::WrapperRegistry::initModule(void)" (?initModule@WrapperRegistry@Pb@@QEAAPEAU_object@@XZ) 함수 어쩌고 저쩌고
```

마찬가지로 pyconfig.h 들어가서 주석처리 하기

```
#ifdef _DEBUG
//#       define Py_DEBUG  
#endif
```

> [참고 글](https://codeday.me/ko/qa/20190827/1370305.html)
  
---

### 4. .obj 파일 : 'x64' 모듈 컴퓨터 종류가 'x86' 대상 컴퓨터 종류와 충돌합니다

시작부터 끝까지 64bit 로 모두 맞추기

1) qt 설치 시 MSVS 버전 맞추고, 64bit 로 설치
2) cmake로 파일 만들때 generator 설정 64bit로 하기
  
---

### 5. manta.exe 실행 시 no module named 'encodings'

```
fatal python error : py_initialize: unable to load the file system codec
modulenotfounderror: no module named 'encodings'
```

빌드할때랑 실행할때의 파이썬 버전 달라서 그런 것.  
cmake 빌드 할 때, 가상환경 아닌 파이썬으로 path 잡기.
(왜인지는 나도 모름....)
