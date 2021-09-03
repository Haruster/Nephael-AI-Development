import pandas as pd # Pandas 라이브러리 사용

pd.Series([1, 2, 3, 4, 5]) # Series는 Pandas에서 제공하는 데이터 타입이며 int가 있는 1차원 배열이다.(문자, 숫자, 논리형 등 모든 데이터 타입을 사용할 수 있음) 데이터 프레임의 한 컬럼들이 다 시리즈이며 왼쪽이 인덱스이며 오른쪽이 값이고 그 값들을 대응하는 것이다.

system_name = ["windows", "mac", "linux", "tmux"] #system_name 변수에 각 이름을 넣는다.
user = [10000, 8000, 6000, 4000] #사용자 수 데이터 지정

result = pd.DataFrame({
	"Name" : system_name,
	"User" : user
}) # 데이터 프레임을 정의함

result # Name과 User가 같이 묶어있는 데이터 프레임을 생성(출력)함

# 데이터 프레임이란 행과 열로 구성된 사각형 모양의 표이다.
