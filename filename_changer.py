import glob
import os
import sys
# file_path = ys.argv[1] #환경변수 1번에서 가져옴.
file_path = './workspace/testset4'
file_list = glob.glob(file_path + '/*.jpg')

# 어떤 번호부터 숫자를 메길지 선택한다.
prefix = input("How to count prefix number?:")
count = 0


for name in file_list:
    # 카운트할 숫자를 1씩 증가시킨다.
    count = count + 1

    print(prefix+str(count))
    name_change = prefix+str(count)
    print(name_change)
    # 파일이름을 변환시킨다.
    os.rename(name, file_path + '/' + name_change + '.jpg')