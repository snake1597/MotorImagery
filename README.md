# MotorImagery
# 摘要 
在世界上有許多患者可能因為意外或年紀增長身體老化而導致行動不便，因此本文提出一種分類運動想像訊號的方法，透過腦波擷取設備結合硬體裝置操控輪椅，輔助行動不便的患者移動。本文著重在如何準確地分辨控制指令與提升操控輪椅時的流暢度。  
  
運動想像(Motor Imagery,MI)是透過想像運動動作產生之腦波，在腦機介面(Brain Computer Interface, BCI)中，可利用這類腦波控制周邊設備的。應用中，如何分類運動想像後的訊號是重要的問題，為了充分利用EEG中序列訊號的時間頻率特徵，本文先將腦波訊號做離散小波轉換(Discrete Wavelet Transform, DWT)提取運動想像腦波的時間頻率特徵，最後透過雙向長短期記憶神經網路(Bidirectional LSTM)識別運動想像訊號。 
# 系統架構    
![image](https://github.com/snake1597/MotorImagery/blob/master/SystemArchitecture.png)
