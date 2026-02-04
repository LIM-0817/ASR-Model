import os
import matplotlib
matplotlib.use('Agg') # GUI 창을 띄우지 않고 파일로만 저장하는 설정
import matplotlib.pyplot as plt
import seaborn as sns
import time

def plot_attention(attention, epoch):
    # 파일 저장할 폴더 없으면 생성
    folder_path = './attention_plot'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"{folder_path} 폴더를 생성함")

    # 시각화
    plt.clf()
    sns.heatmap(attention, cmap='GnBu')
    
    # 저장 경로 설정
    file_name = f'attention_plot_epoch{epoch+1}.png'
    save_path = os.path.join(folder_path, file_name)

    # 저장
    plt.savefig(save_path)
    plt.close()


class TimeElapsed():
    def __init__(self):
        self.start  = -1

    def time_elapsed(self):
        if self.start == -1:
            self.start = time.time()
        else:
            end = time.time() - self.start
            hrs, rem    = divmod(end, 3600)
            min, sec    = divmod(rem, 60)
            min         = min + 60*hrs
            print("Time Elapsed: {:0>2}:{:02}".format(int(min),int(sec)))
            self.start  = -1