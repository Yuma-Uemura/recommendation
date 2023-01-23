import csv
import numpy as np
import math
import pandas as pd



class Item:
    def __init__(self):
        self.limit = 10 #近傍アイテムの数,50％はピアソン上位，50％は両方によって評価された個数が多い順．(提案手法）
        self.piason_user_par=self.limit*(0.5)#追加実験用
        self.piason_zyoui= np.zeros((610,2))#ピアソン造関係数上位(ユーザID,ピアソン相関係数)
        self.onazi_zyoui= np.zeros((610,2))#要素数上位．　　　　　（ユーザID,ピアソン相関係数)
        self.kinbou_user= np.zeros((self.limit))#要素数上位の近傍形成
        self.kinbou_equal=np.zeros((self.limit))#近傍かぶりのユーザ数．確認用

        self.kouho_movie_num=np.zeros((9742))#どちらかに評価された(番号，映画のID,
        self.kouho_movie_user_data=np.zeros((610,3000))#評価された映画の検出用の配列
        self.re_x=0#推薦可能な映画の本数

        ##########################
        self.suisensya_ID=0#推薦される側のユーザID
        self.suisensya_max=0#推薦されるユーザの要素数
        self.yosoku_data=np.zeros((self.limit,2))#欠損させた要素と同じ要素を持つユーザのＩＤと要素番号
        self.user_2=np.zeros((610,3))#userId,(要素数,ave,piason),推薦者のデータを欠損させるためもう一度計算
        self.Pb_k=np.zeros((3000))#予測値
        self.gosa=np.zeros((3000))#MSE
        ##########################

        self.movie_num = 9742 #映画数
        self.user_num = 610 #ユーザー数
        #self.movies = np.zeros((self.movie_num))
        self.user = np.zeros((self.user_num,3))#userId,(要素数,ave,piason)
        #self.user_id_data = np.zeros((100835,4))#要素番号， ユーザID，映画id, 評価値，
        self.user_id_data_2=  np.zeros((611,3000,3))# user_id, 要素番号，(映画id, 評価値)，
        self.table = np.zeros((self.user_num,self.movie_num))
        self.eval_table = np.zeros((self.movie_num,self.limit,2))
        self.piason_data= np.zeros((self.user_num,1))#ピアソン相関係数の値を格納
        self.onazi_youso =  np.zeros((self.user_num,2))#同じ要素を保有する集合と同じ要素を持つ数


        self.init()#ここでピアソンの上位から性能評価のための映画の集合のまとめまでやってしまう．
        self.evaluation()#評価を行う


    def init(self):
        df = pd.read_csv("ratings.csv")

        # 読み込んだ全てのデータを表示します 
        print(df)

        print ( df.at[0,"userId"] )

        print(self.user[0,0])
        self.user[0,0]=df.at[0,"userId"]
        print(self.user[0,0])

        a=df.at[0,"userId"]
        t=0#要素数のカウント
        rate=0#評価の合計
        rate_ave=0

        for num in range(100835):
            if a==df.at[num,"userId"]:
                a=df.at[num,"userId"]
                b=df.at[num,"movieId"]
                c=df.at[num,"rating"]
                
                self.user_id_data_2[a,t,0] = b#映画ID
                
                self.user_id_data_2[a,t,1] = c#rate
                t=t+1
                rate=rate+c

            elif num==10832:#userId,要素数,ave,piason
                rate_ave=rate/t
                self.user[a,0]=t
                self.user[a,1]=rate_ave

            else:  #ユーザーIDが変わったら評価値の平均と要素数をユーザー情報に格納して
                rate_ave=rate/t
                self.user[a,0]=t#最後にユーザＩＤに要素数を書き込み
                self.user[a,1]=rate_ave
                #print(self.user[a,0])
                t=0#要素数初期化
                rate=0#合計初期化
                a=df.at[num,"userId"]
                b=df.at[num,"movieId"]
                c=df.at[num,"rating"]
                self.user_id_data_2[a,t,0] = b#映画ID
                self.user_id_data_2[a,t,1] = c#rate
                t=t+1
                rate=rate+c


            
        #ここからピアソン相関係数の算出を行う．
        #まずどのユーザを対象にした情報推薦なのかを決める．一応要素数が多いユーザを対象にする
        #まず要素数が大きい奴を探す(ユーザID-414)だった
        youso_max=self.user[1,0]
        youso_max_ID=1
        for num in range(610):
            if youso_max<self.user[num,0]:
                youso_max_ID=num
                youso_max=self.user[num,0]

        youso_max=int(youso_max)
        youso_max_ID=int(youso_max_ID)

        print("最大の要素のユーザIDと要素数",youso_max_ID,youso_max)
        # for num in range(3000):
        #     print(self.user_id_data_2[414,num,0],self.user_id_data_2[414,num,1])

        # for num in range(youso_max):
        #     print("414のデータの中身",self.user_id_data_2[youso_max_ID,num,0])

        #ここからピアソン相関係数の計算を行う．ユーザ414に対して行う
        r_1=0
        r_1_1=0
        r_2=0
        r_2_2=0
        a=0
        a_2=0
        onazi_youso_count_num=0
        a_flag=0
        youso_count=0
        youso_range=0
        youso_check=0
        youso_ID=np.zeros((youso_max))
        ave_1=0#推薦される側
        ave_2=0#推薦する側

        #同じ要素をもつ集合とピアソン相関係数の計算

        for num in range(self.user_num):
            youso_range=int(self.user[num,0])
            a=0
            r_1=0
            r_1_1=0
            r_2=0
            r_2_2=0
            
            a_flag=0
            youso_count=0
            youso_ID=np.zeros((youso_max))#平均の計算のためIDを格納
            youso_ID_check=0
            ave_1=0#推薦される側
            ave_2=0#推薦する側
            piason_bunbo=0
            
            
            for num_2 in range(youso_range):
                #print(self.user_id_data_2[num,num_2,0])
                #print("youso_range=",youso_range)
                if self.user_id_data_2[num,num_2,0]==self.user_id_data_2[youso_max_ID,a,0]:
                    # if num!=youso_max_ID:#確認用
                    #     print("------------------------")
                    #     print("syuugou",num,self.user_id_data_2[num,num_2,0])
                    #     print("414",youso_max_ID,self.user_id_data_2[youso_max_ID,a,0])
                    #     print("a=",a)
                    #     print("num_2=",num_2)
                    youso_ID[youso_count]=self.user_id_data_2[num,num_2,0]
                    ave_1=ave_1+self.user_id_data_2[youso_max_ID,a,1]
                    ave_2=ave_2+self.user_id_data_2[num,num_2,1]

                    
                    a_flag=1
                    # r_1=self.user_id_data_2[num,num_2,1]-self.user[num,1]
                    # r_2=self.user_id_data_2[youso_max_ID,a,1]-self.user[youso_max_ID,1]
                    # r_3=r_1*r_2
                    # r_1=r_1*r_1
                    # r_1_1=r_1_1+r_1
                    # r_2=r_2*r_2
                    # r_2_2=r_2_2+r_2
                    # r_3_3=r_3_3+r_3

                    self.kouho_movie_user_data[num,youso_count]=self.user_id_data_2[num,num_2,0]#どの映画IDがかぶったか

                    youso_count=youso_count+1

                    a=a+1

                elif self.user_id_data_2[num,num_2,0]>self.user_id_data_2[youso_max_ID,a,0]:
                    
                    for num_3 in range(youso_range):
                        a=a+1
                        if self.user_id_data_2[num,num_2,0]<=self.user_id_data_2[youso_max_ID,a,0]:
                            break
                        if a>youso_max:
                            break

            if a_flag==0:
                self.user[num,2]=0#同じ要素なしならピアソン相関係数の値は0
            
            elif a_flag==1:
                
                
                #print("piason=",self.user[num,2])
                if youso_count==1:#同じ要素の数が1つの場合はピアソン相関係数は0
                    self.user[num,2]=0

                elif youso_max_ID==num:#自身のピアソン相関係数は0
                    self.user[num,2]=0

                else:
                    ave_1=ave_1/(youso_count)
                    ave_2=ave_2/(youso_count)
                    for num_3 in range(youso_count):
                        youso_ID_check=int(youso_ID[num_3])
                        for num_4 in range(youso_max):
                            if self.user_id_data_2[youso_max_ID,num_4,0]==youso_ID_check:
                                r_1=self.user_id_data_2[youso_max_ID,num_4,1]-ave_1
                            if self.user_id_data_2[num,num_4,0]==youso_ID_check:
                                r_2=self.user_id_data_2[num,num_4,1]-ave_2
                            piason_bunbo=piason_bunbo+(r_1*r_2)
                            r_1_1=r_1_1+(r_1*r_1)
                            r_2_2=r_2_2+(r_2*r_2)

                    if r_1_1==0 or r_2_2==0:
                        self.user[num,2]=0#全ての要素が同じだった場合は0

                    else:
                        self.user[num,2]=piason_bunbo/(math.sqrt(r_1_1)*math.sqrt(r_2_2))#ピアソン相関係数を計算

                
                self.onazi_youso[onazi_youso_count_num,0]=num#同じ要素を保有するユーザID（同じだった場合
                self.onazi_youso[onazi_youso_count_num,1]=youso_count#要素の数
                if num==youso_max_ID:
                    self.onazi_youso[onazi_youso_count_num,1]=0#要素の数
                #print("要素の性質",self.onazi_youso[onazi_youso_count_num,0],self.onazi_youso[onazi_youso_count_num,1])
                onazi_youso_count_num=onazi_youso_count_num+1   #同じ要素を保有するしゅうごうの数
                    
            


        print("同じ要素を持つIDの集合の数=",onazi_youso_count_num)
        #for num in range(onazi_youso_count_num-1):
        #    print("ID,要素数",self.onazi_youso[num,0],self.onazi_youso[num,1])


        #要素の数が多い集合５つ見つける．
        onazi_max=5
        onazi_max_2=onazi_max-1#比較，ソート用
        onazi_count_youso=0
        onazi_count_ID=0
        num_2_2=0
        num_2_3=0
        #flag=0
        for num in range(onazi_youso_count_num):
            if num<onazi_max:
                self.onazi_zyoui[num,0]=self.onazi_youso[num,0]#要素のIDを格納
                self.onazi_zyoui[num,1]=self.onazi_youso[num,1]#要素の要素数
                for num_2 in range(num):
                    num_2_2=num-(num_2+1)
                    num_2_3=num-num_2
                    if self.onazi_zyoui[num_2_2,1]<self.onazi_zyoui[num_2_3,1]:#雑にソート
                        onazi_count_youso=self.onazi_zyoui[num_2_2,1]#要素の数
                        onazi_count_ID =self.onazi_zyoui[num_2_2,0]#要素のID
                        self.onazi_zyoui[num_2_2,1]=self.onazi_zyoui[num_2_3,1]
                        self.onazi_zyoui[num_2_2,0]=self.onazi_zyoui[num_2_3,0]
                        self.onazi_zyoui[num_2_3,1]=onazi_count_youso
                        self.onazi_zyoui[num_2_3,0]=onazi_count_ID
                        
            if num>=onazi_max:

                if self.onazi_youso[num,1]>self.onazi_zyoui[onazi_max_2,1]:#今までの最小と比較し当たらい値が上回ったら入れ替えとソート
                    self.onazi_zyoui[onazi_max_2,1]=self.onazi_youso[num,1]
                    self.onazi_zyoui[onazi_max_2,0]=self.onazi_youso[num,0]

                    for num_3 in range(onazi_max_2):
                        num_2_2=onazi_max_2-(num_3+1)
                        num_2_3=onazi_max_2-num_3
                        if self.onazi_zyoui[num_2_2,1]<self.onazi_zyoui[num_2_3,1]:#雑にソート
                            onazi_count_youso=self.onazi_zyoui[num_2_2,1]#要素の数
                            onazi_count_ID =self.onazi_zyoui[num_2_2,0]#要素のID
                            self.onazi_zyoui[num_2_2,1]=self.onazi_zyoui[num_2_3,1]
                            self.onazi_zyoui[num_2_2,0]=self.onazi_zyoui[num_2_3,0]
                            self.onazi_zyoui[num_2_3,1]=onazi_count_youso
                            self.onazi_zyoui[num_2_3,0]=onazi_count_ID
            

        for num in range(onazi_max):
            print("2つによって評価された要素が多い上位群",num,self.onazi_zyoui[num,0],self.onazi_zyoui[num,1])
        
        #同様に相関係数が大きい順のソートを行う
        piason_hikaku=0
        piason_user_ID=0#ピアソンを参照するための変数
        piason_user=0#ソート用
        piason_num=0#ソート用


        for num in range(onazi_youso_count_num):#両者とも評価を行っている組み
            if self.onazi_youso[num,1]>2:#2乗誤差を求める関係から，ピアソンの近傍は要素数3以上から作成する．
                piason_user_ID=int(self.onazi_youso[num,0])#同じ要素を持つユーザIDを取得
                if self.piason_zyoui[piason_hikaku,1]<self.user[piason_user_ID,2]:
                    self.piason_zyoui[piason_hikaku,0]=piason_user_ID#ユーザID
                    self.piason_zyoui[piason_hikaku,1]=self.user[piason_user_ID,2]#ピアソン相関係数
                    #ソート
                    for num_2 in range(piason_hikaku):
                        num_2_2=piason_hikaku-(num_2+1)
                        num_2_3=piason_hikaku-num_2
                        if self.piason_zyoui[num_2_2,1]<self.piason_zyoui[num_2_3,1]:
                            piason_num=self.piason_zyoui[num_2_2,1]#要素の数
                            piason_user =self.piason_zyoui[num_2_2,0]#要素のID
                            self.piason_zyoui[num_2_2,1]=self.piason_zyoui[num_2_3,1]
                            self.piason_zyoui[num_2_2,0]=self.piason_zyoui[num_2_3,0]
                            self.piason_zyoui[num_2_3,1]=piason_num
                            self.piason_zyoui[num_2_3,0]=piason_user

                    if piason_hikaku<onazi_max_2:
                        piason_hikaku=piason_hikaku+1

        for num in range(onazi_max):
            x=int(self.piason_zyoui[num,0])
            print("ピアソン相関係数の値上位群",self.piason_zyoui[num,0],self.piason_zyoui[num,1],self.user[x,0])
            for num_2 in range(318):
                if self.onazi_youso[num_2,0]==x:
                    print("ピアソン相関係数の同じ数",self.onazi_youso[num_2,1])
        
        
        #近傍形成の集合の作成
        kinbou_par=int(self.limit*0.5)#ピアソン上位50％で近傍形成
        kinbou_check=int(kinbou_par)#近傍ユーザのかぶりのチェック
        kinbou_user_ID=0#かぶったユーザのＩＤ
        kinbou_youso=0

        
        for num in range(self.limit):
            if num<kinbou_par:
                self.kinbou_user[num]=self.piason_zyoui[num,0]#ピアソン上位のユーザIDを格納
                #print(self.kinbou_user[num])
            else:
                self.kinbou_user[kinbou_check]=self.onazi_zyoui[kinbou_youso,0]#要素が多い上位のユーザIDを格納
                for num_2 in range(kinbou_par):#ピアソンと要素のユーザのかぶりがないかのチェック，有るなら格納しない
                    if self.kinbou_user[kinbou_check]==self.kinbou_user[num_2]:
                        self.kinbou_user[kinbou_check]=0
                        kinbou_check=kinbou_check-1
                        self.kinbou_equal[kinbou_user_ID]=self.kinbou_user[num_2]#確認用
                        kinbou_user_ID=kinbou_user_ID+1
                        
                kinbou_check=kinbou_check+1
                kinbou_youso=kinbou_youso+1

        for num in range(self.limit):
            print("近傍形成の集合＝",self.kinbou_user[num])

        self.limit=self.limit-kinbou_user_ID
        print("かぶったユーザ数は", kinbou_user_ID)

        
        
        #提案手法の評価を行うため推薦される側の評価値データにいくつかの欠損データを作成.
        #欠損データと予測値との2乗誤差を求め評価を行う．
        #推薦を行い2乗誤差を求めるにはには少なくとも近傍形成した集合のユーザ1人以上と推薦される側，
        # 両方が評価している必要が有るためそこから求める
        #self.kesson_movie_num
        #self.user_id_data_2=  np.zeros((611,3000,3))# user_id, 要素番号，(映画id, 評価値)，
        #self.kouho_movie_num=np.zeros((9742,2))#どちらかに評価された映画のIDと要素数

        movie_user_ID_check=0
        movie_kinbou_youso_num=0
        movie_youso_num=0
        
        re_x_2=0
        re_y=0#評価した人数カウント
        flag_movie=0

        print(youso_max)

        
        for num_2 in range(self.limit):
            movie_user_ID_check=int(self.kinbou_user[num_2])#近傍ユーザのID番号
            for num_4 in range(self.user_num):
                if movie_user_ID_check==self.onazi_youso[num_4,0]:
                    movie_kinbou_youso_num=int(self.onazi_youso[num_4,1])
            
            print("movie_user_ID_check",movie_user_ID_check,movie_kinbou_youso_num)

            for num_3 in range(movie_kinbou_youso_num):#近傍のユーザと推薦される側の映画IDチェック．
                
                self.kouho_movie_num[self.re_x]=self.kouho_movie_user_data[movie_user_ID_check,num_3]#映画Id

                for num in range(self.re_x):
                    #今までに追加した映画にかぶりが有るなら
                    if self.kouho_movie_num[num]==self.kouho_movie_user_data[movie_user_ID_check,num_3]:
                        self.kouho_movie_num[self.re_x]=0
                        re_y=re_y+1
                        self.re_x=self.re_x-1
                        break
                self.re_x=self.re_x+1




        print("要素の合計(かぶり無し)=",self.re_x)
        print("同じ要素=",re_y)
        #評価を行うための準備
        self.suisensya_ID=youso_max_ID
        self.suisensya_max=youso_max

        
        #for num in range(movie_youso_num):
            # print(self.kouho_movie_num[movie_youso_num,0],self.kouho_movie_num[movie_youso_num,1])
        
    def evaluation(self):
        print("evaluation")
        print("提案手法の平均二乗誤差(MSE)の算出を行う")
        print("近傍形成の要素は")
        for num in range(self.limit):
            print(self.kinbou_user[num])
        print("である")
        print('推薦可能な映画の本数は',self.re_x)
        print("これら全ての要素に対して1要素ごとに欠損データを作成し予測値を算出しMSEで評価を行う")

        No_movie_ID=0
        erased_ID=0
        erased_ID_youso=0
        erased_eva=0

        for erase in range(self.re_x):
        #for erase in range(1):

            No_movie_ID=self.kouho_movie_num[erase]
            #print("消えた映画",No_movie_ID)
            #消えた映画をself.user_id_data_2から削除
            for num in range(self.suisensya_max):
                if  self.user_id_data_2[self.suisensya_ID,num,0]==No_movie_ID:
                    erased_ID=No_movie_ID
                    erased_ID_youso=num
                    erased_eva=self.user_id_data_2[self.suisensya_ID,num,1]
                    #print("kesson",self.user_id_data_2[self.suisensya_ID,num,0])
                    self.user_id_data_2[self.suisensya_ID,num,0]=0#欠損させる．
                    #print("kesson",self.user_id_data_2[self.suisensya_ID,num,0])
            
            #映画を欠損させたためもう一度平均の算出を行う．
            ave=0
            for num in range(self.suisensya_max):
                if self.user_id_data_2[self.suisensya_ID,num,0]!=No_movie_ID:
                    ave=ave+self.user_id_data_2[self.suisensya_ID,num,1]
            ave=ave/(self.suisensya_max-1)



            #欠損した状態からピアソン相関係数を算出
            r_1=0
            r_1_1=0
            r_2=0
            r_2_2=0
            a=0
            a_flag=0
            youso_count=0
            youso_range=0
            youso_range_ID=0
            youso_check=0

            #欠損したデータと同じ要素をもつ集合をもとめ，
            # ピアソン相関係数と類似度を計算
            t_1=0

            for num in range(self.limit):
                youso_range_ID=int(self.kinbou_user[num])
                youso_range=int(self.user[youso_range_ID,0])
                for num_2 in range(youso_range):
                    if self.user_id_data_2[youso_range_ID,num_2,0]==erased_ID:#欠損させた映画IDと同じ要素を保有するなら
                        self.yosoku_data[t_1,0]=youso_range_ID#配列にIDを保存
                        self.yosoku_data[t_1,1]=num_2

                        t_1=t_1+1
                        
                        #print("kabutterukazu",t_1)
                        break
            #print("erased_ID=",erased_ID)
            get_piason_ID=0

            a=0
            r_1=0
            r_1_1=0
            r_2=0
            r_2_2=0
            a_flag=0
            youso_count=0
            suisen_youso_ID=np.zeros((self.suisensya_max))#平均の計算のためIDを格納
            youso_ID_check=0
            ave_1=0#推薦される側
            ave_2=0#推薦する側
            piason_bunbo=0
            youso_num=0
            

            
            for num in range(t_1):#かぶった要素に対してピアソン相関係数を算出
                #初期化
                r_1=0
                r_1_1=0
                r_2=0
                r_2_2=0
                a_flag=0
                youso_count=0
                suisen_youso_ID=np.zeros((self.suisensya_max))#同じ要素の映画ID
                youso_ID_check=0
                ave_1=0#推薦される側
                ave_2=0#推薦する側
                piason_bunbo=0
                
                #t_1のIDを取得
                get_piason_ID=int(self.yosoku_data[num,0])
                a=0
                for num_2 in range(self.suisensya_max):
                    if self.user_id_data_2[get_piason_ID,num_2,0]==self.user_id_data_2[self.suisensya_ID,a,0]:

                        suisen_youso_ID[youso_count]=self.user_id_data_2[get_piason_ID,num_2,0]

                        ave_1=ave_1+self.user_id_data_2[self.suisensya_ID,a,1]
                        ave_2=ave_2+self.user_id_data_2[get_piason_ID,num_2,1]


                        a_flag=1
                        

                        youso_count=youso_count+1
                        a=a+1

                    elif self.user_id_data_2[get_piason_ID,num_2,0]>self.user_id_data_2[self.suisensya_ID,a,0]:
                    
                        for num_3 in range(self.suisensya_max):
                            a=a+1
                            if self.user_id_data_2[get_piason_ID,num_2,0]<=self.user_id_data_2[self.suisensya_ID,a,0]:
                                break
                            if a>self.suisensya_max:
                                break

                if a_flag==0:
                    self.yosoku_data[num,1]=0#同じ要素なしならピアソン相関係数の値は0
                
                elif a_flag==1:
                    #print("推薦者のIDと要素数",get_piason_ID,youso_count)
                    
                    
                    #print("piason=",self.user[num,2])
                    if youso_count==1:#同じ要素の数が1つの場合はピアソン相関係数は0
                        self.yosoku_data[num,1]=0

                    
                    else:
                        ave_1=ave_1/(youso_count)
                        ave_2=ave_2/(youso_count)
                        for num_3 in range(youso_count):
                            youso_ID_check=int(suisen_youso_ID[num_3])
                            for num_4 in range(self.suisensya_max):
                                if self.user_id_data_2[self.suisensya_ID,num_4,0]==youso_ID_check:
                                    r_1=self.user_id_data_2[self.suisensya_ID,num_4,1]-ave_1
                                if self.user_id_data_2[get_piason_ID,num_4,0]==youso_ID_check:
                                    r_2=self.user_id_data_2[get_piason_ID,num_4,1]-ave_2
                                piason_bunbo=piason_bunbo+(r_1*r_2)
                                r_1_1=r_1_1+(r_1*r_1)
                                r_2_2=r_2_2+(r_2*r_2)
                        
                        if r_1_1==0 or r_2_2==0:#要素が同じ
                            #print("評価：分母が0")
                            self.user_2[num,0]=get_piason_ID
                            self.user_2[num,2]=0
                        else:
                            self.user_2[num,0]=get_piason_ID
                            self.user_2[num,2]=piason_bunbo/(math.sqrt(r_1_1)*math.sqrt(r_2_2))#ピアソン相関係数を計算
                        youso_num=youso_num+1

            # for num in range(t_1):
            #     print("欠損データと同じ要素を持つユーザとピアソン相関係数",self.user_2[num,0],self.user_2[num,2])

            #欠損データに対して嗜好の予測値の計算
            bunbo=0
            bunsi=0
            sikou_ID=0
            sikou_youso_data=0
            for num in range(t_1):
                sikou_ID=int(self.user_2[num,0])
                sikou_youso_data=int(self.yosoku_data[num,1])
                bunsi=bunsi+(self.user_id_data_2[sikou_ID,sikou_youso_data,1]-self.user[sikou_ID,1])*self.user_2[num,2]
                if self.user_2[num,2]>0:
                    bunbo=bunbo+self.user_2[num,2]
                else:
                    bunbo=bunbo+(self.user_2[num,2]*-1)

            #print("ave=",ave)
            #print("bunbo=",bunbo)
            #print("bunsi=",bunsi)
            self.Pb_k[erase]=ave+(bunsi/bunbo)
            #print(self.Pb_k[erase])

            self.gosa[erase]=(erased_eva-self.Pb_k[erase])*(erased_eva-self.Pb_k[erase])#2乗誤差
            #最後に消したデータを元にに戻す
            self.user_id_data_2[self.suisensya_ID,erased_ID_youso,0]=No_movie_ID
            print("進行度=%d/%d" % (erase+1,self.re_x))

        MSE=0
        for num in range(self.re_x):
            MSE=MSE+self.gosa[num]
        MSE=MSE/self.re_x
        print("MSE=",MSE)



            
            

                
                    
            
            



        

        
                    





            


        




                    




            




if __name__ == "__main__":
    Item()