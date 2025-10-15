# 一.环境搭建
![alt text](<img/截屏2025-10-10 19.19.12.png>)
# 二.实现一个简单的模型
## 1.baseline_net
![alt text](<img/截屏2025-10-11 21.27.55.png>)
## 2.深层网络，梯度消失、训练困难
### 残差网络
方法之一就是用残差网络，不过在part1对于这一点，我自认为已经做了一个十分详尽的阐述了，这里就不再重复了
### 空洞卷积
![alt text](<img/截屏2025-10-10 20.17.13.png>)
空洞卷积是在卷积核的采样点之间插入空格

也就是说

它在不增加卷积核参数数量的情况下，让卷积核“覆盖”更大的输入区域，也就是扩大了感受野

R=k+(k−1)(r−1)这个感受野的计算公式也是显然易见的

### 多尺度特征融合
输入同一特征图，然后分别用不同感受野的卷积提取特征（即不同kernel或dilation最后拼接或相加融合）    
这样网络就短了很多   
如图所示    
![alt text](<img/截屏2025-10-10 20.49.32.png>)

按照我的理解就是，普通的卷积层就是靠逐层卷积来逐步从局部特征到全局特征，逐步扩大感受野

>这是“递推”思想    
而多尺度特征融合是“分治”！

想学全局特征，得以局部特征为基础，慢慢扩展到全局     
但现在不逐步学     
现在我分别去各种的尺度上进行学习，然后再“融合”    
或者说     
单独的大尺度卷积构建图像的“结构骨架”，负责全局感知     
然后小尺度卷积填充细节、补充纹理，使得结构生动且真实

# 三.这个代码是什么意思？
## 1.运行结果
![alt text](<img/截屏2025-10-11 08.05.12.png>)
![alt text](<img/截屏2025-10-11 17.24.40.png>)
***
## 2.代码新知识！😊
### 2.1 命令行参数运行
#### 2.1.1 命令行参数配置函数
这个函数get_args()的作用就是：     
用argparse模块定义并解析命令行参数     
让程序在运行时可以灵活地改变训练行为（比如模型类型、学习率、损失函数权重等）
- ```argparse.ArgumentParser()```：创建一个命令行参数解析器
- ```description```：是程序的简介，在执行 ```python main.py -h```时会显示
- ```add_argument()```：逐个定义你想在命令行中传入的参数
- ```parser.parse_args()```：把命令行里输入的参数提取成一个对象（通常命名为 args）
比如     
```
args.data_dir == "./dataset"
args.model == "unet"
args.mode == "train"
```
##### 具体解读
如对于    
```
parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset")
```
```--data_dir```表示数据集路径，参数名为 --data_dir
```required=True```表示这个参数必须提供，否则会报错
```type=str```表示它的数据类型是字符串
```help="Path to the dataset"```这个 help参数本身不会影响程序运行，它是给用户自动生成命令行帮助说明的
可通过```python main.py -h```      
输出自动生成的帮助文档，其中就会自动生成这个说明
***
### 2.2 DerainDataset
#### 1.根据模式生成对应文件夹路径
![alt text](<img/截屏2025-10-11 15.05.43.png>)     
用join（）进行拼接😋     
它的核心优势是能自动处理不同操作系统下路径分隔符的差异（如 Windows 用 \，Linux/macOS 用 /），让代码在不同系统上都能正常运行     
#### 2.将input和target文件下的图片的文件名进行排序
这样保证了input和target文件名一一对应✌️
>不过，为什么只对 self.rainy_dir 排序🤯而没有对 self.clean_dir 排序呢？😭       
![alt text](<img/截屏2025-10-11 15.40.43.png>)
      
clean_path是通过rainy_images的同名文件名构造的     
也就是说，clean_dir不需要再排序，因为它完全依赖rainy_images的顺序来访问     
所以两个文件夹结构是一一对应的     
每个带雨图像在 input/      
它的无雨真值图像在 target/ 下    
并且文件名完全一致    
#### 3.预处理
分模式进行
#### 4.魔法方法😋
出现了方法名，像__len__这样前后带有"_""_"是什么意思？🤔    
这些函数在 Python 里有一个专门的名字：    
它们叫做 “魔术方法” 或 “双下划线方法”，这些方法是 Python 内部约定的特殊钩子函数       
当执行某些操作时（比如 len(obj)、obj[0]、print(obj)）      
Python 会自动调用这些对应的“魔术方法”     
你不需要手动去调用它们（.函数名）
Python 解释器会在合适的时候自己触发
![alt text](<img/截屏2025-10-11 15.20.13.png>)
#### __getitem__方法
可以```rainy, clean, name = train_dataset[0]```手动调用
或者让DataLoader自动调用   
首先拼接完整的图片的路径    
然后打开图像并转成RGB    
最后应用transform预处理再返回     
![alt text](<img/截屏2025-10-11 15.42.52.png>)     
#### 和普通的dataset的区别
为什么要自定义呢？    
之前加载图片用的是datasets.ImageFolder       
不妨来对比一下😋

##### 定位区别
![alt text](<img/截屏2025-10-11 15.51.33.png>)
##### 一对一配对
去雨、去噪、超分辨率等任务中，我们不是要输出一个“标签”    
而是要从“输入图像”生成另一个“目标图像”    
ImageFolder无法表达这种一对一关系，所以就手动实现
##### 灵活性
显然，这样的自定义可以对不同模式，文件结构，返回内容等作出任何的自定义和处理，具有灵活性
***
### 2.3 train_one_epoch
#### 1.tqdm
用来进行每一轮的训练

之前我们是用```for imgs, targets in train_loader```     
每次取出一个batch    
现在我们用    

```
loop = tqdm(loader, leave=True)
    for _, (rainy, clean, _) in enumerate(loop):
```
tqdm 是一个进度条库，会让训练时在终端实时显示进度    
loader 是 DataLoader，每次会返回一个 batch 的训练样本    
所以 tqdm(loader) 相当于“带进度条的 DataLoader 循环”     
leave=True，保留进度条，显示每个 epoch 的结果     
leave=False，进度条消失，只显示最新的一个

#### 2.enumerate
这里enumerate似乎好像完全没发挥什么作用     
可能是我们想要索引的时候可以方便我们简单的更改吧     
然后后面的就是常规操作了
***
### 2.4 best_psnr与保存
如果遇到更好的```current_psnr > best_psnr```
#### 参数打包
用```checkpoint_data = {"state_dict": ..., "optimizer": ...}```
把模型参数张量（卷积核、偏置等）和优化器内部状态（如 Adam 的一阶/二阶动量）都打包

>为啥要存优化器？🤔

 如果你中断后续训练并从这个 best 继续，优化器也要恢复到同样状态，训练轨迹才连贯


#### 命名
根据有没有用感知损失用```model_name = f"{args.model}_perceptual..." if args.use_perceptual_loss```分别进行命名

#### 保存
save_checkpoint是定义在utils.py里的方法

（感觉定义在utils.py都是比较“隐蔽”的，不是那种大的，结构性的方法）
用```save_checkpoint(checkpoint_data, filename=f"best_{model_name}")```
具体是用torch.save(state, f"{folder}/{filename}")

于是项目文件夹里就会多一个checkpoint的文件夹，里面有一个名为filename的文件

### 2.5 可视化结果
```save_some_examples(model, test_loader, epoch, folder="evaluation_images", device=device)```
每训练完一轮（epoch），从测试集里取出几张图片，    
把「原图（带雨）」→「模型去雨结果」→「真实无雨图」拼接起来保存成一张图，      
存到 evaluation_images/ 文件夹中，方便你对比查看模型效果变化

#### 取一小批
```iter(loader)```，把数据加载器这个可迭代对象转为一个迭代器

```next()```从中取出第一批batch

#### 拼接
torch.no_grad()，然后将rainy, derained, clean进行纵向拼接，最后保存

还有，如果指定的文件夹不存在那就创建一个

### 2.6 test与evaluate
test时有两次加载

第一次```torch.load（args.checkpoint）```
把整个文件读进内存，得到一个字典对象
=
第二次用```model.load_state_dict()``` 
``` optimizer.load_state_dict()```

把权重和参数分别加载到模型和优化器里

#### evaluate
就是对每个batch计算指标，然后除以len(loader)来计算平均

最后return avg_psnr，返回指标用于判断最优模型

### 2.7 calculate_metrics
负责具体计算avg_psnr, avg_ssim, lpips_score
- 类型转换

```
img1_np = img1.detach().cpu().numpy().transpose(0, 2, 3, 1)
img2_np = img2.detach().cpu().numpy().transpose(0, 2, 3, 1)
```
![alt text](<img/截屏2025-10-11 19.34.47.png>)
- 限制像素范围 [0,1]
```
img1_np = np.clip(img1_np, 0, 1)
img2_np = np.clip(img2_np, 0, 1)
```
训练中网络输出可能略超 1（如 1.01）

或略小于 0（如 -0.02）

np.clip() 会强制截断到 [0,1] 区间，保证计算指标时合法

- 循环计算
```
batch_psnr, batch_ssim = 0, 0
for i in range(img1_np.shape[0]):
    batch_psnr += psnr(img1_np[i], img2_np[i], data_range=1.0)
    batch_ssim += ssim(img1_np[i], img2_np[i], data_range=1.0, multichannel=True, channel_axis=2, win_size=7)
```
对batch 内的每一张图做单独计算

- 计算平均
```
avg_psnr = batch_psnr / img1_np.shape[0]
avg_ssim = batch_ssim / img1_np.shape[0]
```


