---
layout: post
title: Deep Fluids
comments: true
tags : [graphics, deeplearning]
---
# Deep Fluids: A Generative Network for Parameterized Fluid Simulations
> Byungsoo Kim et al.


## 필수 Background

- 오토인코더(Auto Encoder)

  [<https://wikidocs.net/3413>](https://wikidocs.net/3413)

  - Stacked denoising autoencoders
  - Sparsity

- Latent space

- Parametrization






## Abstract

### Figure1

훈련을 위한 입력 시뮬레이션 + 생성을 위한 몇가지 파라미터 => fluid의 속도를 연속적으로 합성

-> 이 모델은, 속도의 재구성, 연속적인 보간, 잠재 공간 시뮬레이션을 빠르게 할 수 있음



- reduced parameter 로부터 fluid 시뮬레이션을 합성하는 generative 모델
- convolution network는 속도장(velocity field)에 대해 학습함 (속도장 : discrete, parameterizable)
- 새로운 loss function : divergence-free 속도장을 보장함
- reduced space 에서 복잡한 parameterizations 다루기 가능
- latent space에서 두번째 네트워크를 통합하여, 시뮬레이션을 제 시간에 진행 가능 **(????????)**
- 다양한 응용 분야 : 시뮬레이션의 빠른 구성, 다양한 매개변수를 사용한 fluids 보간, 시간 re-샘플링, latent space 시뮬레이션, fluid 시뮬레이션 데이터 압축 등.
- 합성된 속도장 : CPU solver 사용해서 재 시뮬레이션 하여 얻는 것 보다 700 배 빠르고, 1300배의 압축률.



## 1. Introduction

- reduced parameters 로부터 dynamic Eulerian fluids 시뮬레이션 속도를 완전히 구성하는 첫 생성 모델 네트워크이다.

- 다양한 fluids의 움직임 가능 : 난류 연기(turbulent smoke) 에서 끈적한 액체(gooey liquids) 까지
- 학습 된 fluids 상태를 효율적으로 복구 및 훈련 데이터와 직접적으로 일치하지 않는 입력 매개 변수에 대해서도 적절한 속도 장 생성
- 연구의 기술적 기여
  - 첫 딥러닝 생성 모델 : reduced parameters 로부터 2D, 3D fluids 시뮬레이션 속도 합성
  - parameterizable 속도 장을 정확하게 인코딩 하는 fluids 생성 모델 : 1300배의 압축률, 700배의 성능 속도 향상
  - 오토인코더 구조를 통해 시뮬레이션 클래스들을 latent space representation으로 인코딩 하는 접근 방식 :  latent space 통합 네트워크와 결합하여 **시간을 앞당기고(?)** fluid 시뮬레이션과의 유연한 인터렉션 가능하게 한다.
  - 훈련 데이터와 직접적 연관이 있고, parameter space의 중간 지점인 샘플들을 재구성 할때 제안 된 방법에 대한 자세한 분석.



## 3. A Generative Model For Fluids

- 목표 : 원래의 속도장 데이터 셋을 근사하는 CNN 훈련시키는 것

  -> CNN : loss function 최소화 함으로써, data manifold를 shift-invariant feature map으로 구성함

- 입력값 : 매개변수화 가능한(parameterizable) 데이터 셋



### 3.1 Loss Function for Velocity Reconstruction

- 네트워크 입력 : $$[ \bold u_c, \bold c ]$$

  - $$\bold u_c \in \mathbb{R}^{H\times W\times D \times V_{dim}}$$ : single velocity vector field frame in $$V_{dim}$$, height $$H$$, width $$W$$, depth $$D$$(1 for 2D)

  - $$\bold c = [c_1, c_2, \cdots , c_n] \in \mathbb{R}^n$$ : solver's parameters 

  ​	ex. 2D : $$c$$ = $$x$$ position, width(of smoke source), time(of frame) 의 combination

  ​	나비에 스톡스 방정식의 비선형성 때문에, 이 세 parameters들은 매우 다른 속도 셋을 출력

  

- 새로운 손실 함수

  - 유체역학 : 질량 보존(비압축성 흐름을 위한 divergence-free 모션을 보장하는 것)이 매우 중요함
  - $$L_G(\bold c) = \lVert \bold u_c - \nabla \times G(\bold c) \rVert_1$$
    - $$G(\bold c) : \mathbb{R}^n \mapsto \mathbb{R}^{H\times W\times D \times G_{dim}} $$  : 네트워크 output
    - $$\nabla \times G(\bold c)$$ : reconstruction 타겟, divergence-free 보장됨
    - 비압축성 유동(divergence-free)에 적합
  - $$L_G(\bold c) = \lVert \bold u_c - G(\bold c) \rVert_1$$
    - 부분적으로 발산하는 모션 : 직접적 속도 추론이 더 좋은 근사임(curl 삭제)
  - $$L_G(\bold c) = \lambda_\bold u \lVert \bold{u_c - \hat{u}_c} \rVert_1 + \lambda_{\nabla \bold u} \lVert \nabla\bold{u_c} - \nabla\bold{\hat{u}_c}\rVert_1$$ 
    - $$\hat{u}_c = \nabla \times G(\bold c)$$ for 비압축성 유동
    - $$\hat{u}_c = G(\bold c)$$ for 압축성 유동
  - $L_G(\bold c) = \lambda_\bold u \lVert \bold{u_c - \hat{u}_c} \rVert_1 + \lambda_{\nabla \bold u} \lVert \nabla\bold{u_c} - \nabla\bold{\hat{u}_c}\rVert_1$ 
    - $\hat{u}_c = \nabla \times G(\bold c)$ for 비압축성 유동
    - $\hat{u}_c = G(\bold c)$ for 압축성 유동


### 3.2 Implementation

- 구조 : BEGAN 구조 차용

- 초기 파라미터 $$\bold c$$  -> $$m$$-dim 가중치 벡터 $$\bold m$$ (by fc 레이어)

- <img src="/img/deepfluids1.png" width="600px">

  

## 4. Extended Parameterizations

- 매개변수 많은 장면은, 매개변수화 하기 어려울 수 있음

  i.e. $$[\bold p_0, \bold p_1, \cdots, \bold p_t] \rightarrow \bold u_t$$ 

  ​	$$\bold p_t$$ : smoke source position

  ​	$$\bold u_t$$ : reconstructed velocity field at time $$t$$

- 프레임 수에 따라 매개변수 수가 선형적으로 늘어남 -> 매개변수 공간 : 데이터 중심의 접근 방식 불가능

- 따라서 $$G^\dagger(\bold u) : \mathbb{R}^{H\times W\times D \times V_{dim}} \mapsto \mathbb{R}^n$$ (인코더 구조)를 generator에 추가
  - 시간 통합을 위해 두 번 째 작은 네트워크와 결합
  - velocity field frames --> parameterizations $$\bold c = [\bold{z, p}] \in \mathbb{R}^n$$    : 매핑시킴
    - $$\bold z \in \mathbb{R}^{n-k}$$ : reduced latent space(=flow의 임의의 features를 만드는 in unsupervised way) (latent vector)

    - $$\bold p \in \mathbb{R}^k$$ : supervised parameterization to control specific attributes (control vector)

    - separation은 훈련하는 동안 latent space를 sparse 하게 만듬 -> reconstruction quality 증가 **~~(왜??)~~**

      > i.e. moving smoke source example : $$n=16$$ & $$\bold p$$ encodes $$x, z$$ positions 

- $$L_{AE}(\bold u) = \lambda_\bold u \lVert \bold{u_c - \hat{u}_c} \rVert_1 + \lambda_{\nabla \bold u} \lVert \nabla\bold{u_c} - \nabla\bold{{\hat u}_c}\rVert_1 + \lambda_\bold p \lVert \bold p - \bold{\hat p} \rVert^2_2$$ 

  - $$\hat{\bold p}$$ : part of the latent space vector constrained to represent control parameters $$\bold p$$
  - 속도 장의 상태가 나머지 latent space의 dim $$\bold z$$ 로 표시되므로 복잡한 매개변수화 처리 가능
  - 시간 차원을 매개변수로 explicitly 하게 인코딩 하지 않는 latent space 사용 가능
  - 대신에, 적절한 latent codes의 sequence를 생성하는 두 번째 latent space integration network 사용



### 4.1 Latent Space Integration Network

- latent space : velocity field states $$\bold z$$ 에 의해 오직 시간의 확산 표현만 학습
- reduced representation 로부터 시간을 앞당기는 latent space integration network 제안

<img src="/img/deepfluids2.png" width="500px">

- $$T(\bold x_t) : \mathbb{R}^{n+k} \mapsto \mathbb{R}^{n-k}$$
  - $$\bold x_t = [\bold{c_t} ; \Delta\bold{p_t}] \in \mathbb{R}^{n+k}$$ : input vector (concatenation of $$\bold c_t$$ and $$\Delta \bold p_t$$)
    - $$\bold c_t$$ : latent code at current time t
    - $$\Delta\bold{p_t}=\bold p_{t+1} - \bold p_t \in \mathbb{R}^k$$ : control vector difference bet user input parameters
  - output of $$T(\bold x_t)$$ : residual $$\Delta \bold z_t$$ bet two consecutive states

    - $$\bold z_{t+1} = \bold z_t + T(\bold x_t)$$ : new latent code
  - $$T$$ : multilayer perceptron network (3 FC layer + ELU)
    - navigator on the manifold of the latent space
    - controlled individual steps (rather than physically induced)
- $$L_T(\bold x_t, \cdots, \bold x_{t+w-1}) = \frac{1}{w} \sum_{i=t}^{t+w-1} \lVert \Delta \bold z_i - T_i \rVert_2^2$$
  - window of $$w$$ sequential latent codes with an $$L_2$$ loss function

  - $$T_i$$ : recursively computed from $$t$$ to $$i$$ 

    

<img src="/img/deepfluids3.png" width="500px" />

  - 순서

    - initial incompressible velocity field($$\bold{u_0}$$) -> initial reduced space($$\bold{c_0}$$) by $$G^\dagger(\bold{c_0})$$ 

    - concatenating : reduced space($$\bold c_t$$) + position update($$\Delta \bold p_t$$) -> $$\bold x_t$$

    - compute $$\bold z_{t+1} $$ by latent space integration network $$T$$  ($$\bold z_{t+1} = \bold z_t + T(\bold x_t)$$ )

    - compute $$\bold c_{t+1} = [\bold z_{t+1};\bold p_{t+1}]$$ 

    - reconstruct velocity field $$\bold u_{t+1}$$ by $$G(\bold c_{t+1})$$ 

      

## 5. Results

- velocity fields - smoke simulations / advect densities - surfaces for liquids



### 5.1 2-D Smoke Plume

- training set : combination of 

  5 samples with varying source widths $$w$$  &

  21 samples with varying $$x$$ positions $$\bold p_x$$ 

- each simulations : 200 frames / grid resolutions 96 x 128 / domain size (1, $$1.\overline{33}$$)
- trained total 21,000 unique velocity field samples
- 

### 5.2 3-D Smoke Examples

