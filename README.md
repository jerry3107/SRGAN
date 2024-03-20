# SRGAN實現影像放大
此次專題是透過Tensorflow來復刻"_Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network_"這篇論文中的神經網路架構並進行訓練。下面為使用訓練後的模型及使用Bicubic進行圖片放大4倍的比較結果:

![bicubic](https://github.com/jerry3107/SRGAN/assets/105486398/1fc4db37-f586-4eb9-b514-8b08ba4df46c) **Bicubic**  
![SR](https://github.com/jerry3107/SRGAN/assets/105486398/a4c29aa2-218d-4dd7-af5e-314eb8ff1013)**SRGAN**  
![HR](https://github.com/jerry3107/SRGAN/assets/105486398/0f131849-00ad-4d14-ae25-1b3d37784597)**原圖**  
可以看出相較於Bicubic放大，透過SRGAN的模型放大的結果有更好的解析度

