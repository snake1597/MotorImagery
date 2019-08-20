# MotorImagery
使用離散小波轉換與長短期記憶神經網絡實現運動想像的腦機介面系統
摘要
運動想像(Motor Imagery,MI)是透過想像運動動作產生之腦波，在腦機介面(Brain Computer Interface, BCI)中，可利用這類腦波控制周邊設備的。應用中，如何分類運動想像後的訊號是重要的問題，為了充分利用EEG中序列訊號的時間頻率特徵，本文使用長短期記憶神經網路(Long Short-Term Memory, LSTM)的變體雙向長短期記憶神經網路(Bidirectional LSTM)來進行訊號的分類。本篇自行錄製了四種腦波訊號，分別為想像左手、想像右手、想像舌頭與眨眼動作，本篇首先利用g.SAHARAsys系統結合Emotiv EPOC IC擷取腦波訊號，並以無線方式接收並傳輸到電腦。根據國際10-20系統，我們使用C3、C4及Cz等三個電極點來截取腦波訊號，接著以離散小波轉換(Discrete Wavelet Transform, DWT)提取運動想像腦波的時間頻率特徵。最後透過雙向長短期記憶神經網路(Bidirectional LSTM)識別運動想像訊號。
關鍵字：腦機介面、離散小波轉換、運動想像、遞歸神經網絡、雙向長短期記憶
