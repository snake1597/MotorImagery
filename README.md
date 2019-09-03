# MotorImagery
# 摘要 
&emsp;&emsp;在世界上有許多患者可能因為意外或年紀增長身體老化而導致行動不便，因此本文提出一種分類運動想像訊號的方法，透過非侵入式腦波擷取設備結合硬體裝置操控輪椅，輔助行動不便的患者移動。本文著重在如何準確地分辨控制指令與提升操控輪椅時的流暢度。  
  
&emsp;&emsp;運動想像(Motor Imagery,MI)是透過想像運動動作產生之腦波，在腦機介面(Brain Computer Interface, BCI)中，可利用這類腦波控制周邊設備的。應用中，如何分類運動想像後的訊號是重要的問題，為了充分利用EEG中序列訊號的時間頻率特徵，本文先將腦波訊號做離散小波轉換(Discrete Wavelet Transform, DWT)提取運動想像腦波的時間頻率特徵，最後透過雙向長短期記憶神經網路(Bidirectional LSTM)識別運動想像訊號。 
# 系統架構    
![image](https://github.com/snake1597/MotorImagery/blob/master/SystemArchitecture.png)  
&emsp;&emsp;首先，透過 g.SAHARsys乾式電極系統擷取腦波訊號並傳送給 EPOC chip，再經由無線的方式傳送給Jetson TK1 識別腦波訊號，識別的過程先將訊號經過 DWT 提取各類別的特徵值，最後再透過雙向長短期記憶來識別訊號，Jetson TK1 使用藍芽與 FPGA 上的藍芽連線配對，再將所判斷出來的類別透過藍芽傳送不同的控制命令來驅動電動輪椅移動，如圖 25 為電動輪椅控制系統架構圖，本系統設計連續兩秒分類為相同類別，才會讓電動輪椅動作。  
# 開發環境
Programming Language: Python  
IDE: Visual Studio Code
# 程式說明  
搭建模型後進行訓練與驗證，並保存最佳的模型進行測試，還有顯示訓練中accuracy與loss的曲線。
<div><pre>python Bidirectional_LSTM.py</pre></div>  
資料前處理，採用DWT。
<div><pre>python DWT.py</pre></div>  
載入訓練資料。
<div><pre>python loadData.py</pre></div>  
顯示資料處理前跟處理後。  
<div><pre>python showData.py</pre></div>  
# 備註  
&emsp;&emsp;本篇只示範資料預處理與訓練網路的過程，資料庫以BCI Competition II裡的Data set III當作訓練神經網路的資料，訓練資料跟測試資料個為140組，類別為想像左手跟想像右手(前者代號為0，後者為1)  
Slide Link: https://drive.google.com/file/d/1WHMUEh1P3lko4vO81LceTVSXd3wgi3Pa/view
