#### 第一章 预备知识

##### 一、Python 基础

###### 1. 列表推导式和条件赋值

```python
# 生成一个数字序列
L = []
def my_func(x):
    return 2*x
for i in range(5):
    L.append(my_func(i))

# 用列表推导式简化写法
[my_func(i) for i in range(5)]

# 列表推导式支持多层嵌套
[m + '_' + n for m in ['a', 'b'] for n in ['c', 'd']]

# 带 if 的条件赋值
a, b = 'cat', 'dog'
condition = 2 > 1  # True
if condition:
    value = a
else:
    value = b

# 条件赋值简化写法
value = 'cat' if 2>1 else 'dog'
```

###### 2. 匿名函数与 map 方法

```python
# lambda 用于映射关系
my_func = lambda x: 2*x
multi_para_lambda = a, b: a + b

# lambda 函数在无需多出调用的场合使用更频繁
[(lambda x: 2x)(i) for i in range(5)]

# 对于上述这种列表推导式的匿名函数映射，python 中提供了 map 函数来完成，返回的是一个 map 对象，需要通过 list 转为列表
# map()是 Python 内置的高阶函数，它接收一个函数 f 和一个 list，并通过把函数 f 依次作用在 list 的每个元素上，得到一个新的 list 并返回
map(f,list)
list(map(lambda x: 2*x, range(5)))
list(map(lambda x: y: str(x)+'_'+y, range(5), list('abcde')))
```

###### 3. zip 对象和 enumerate 方法

```python
# zip 函数可以把多个可迭代对象打包成一个元组构成的可迭代对象，返回的是一个 zip 对象，可以通过 tuple, list 得到相应结果
L1, L2, L3 = list('abc'), list('def'), list('hij')
list(zip(L1, L2, L3)) 
# 输出：[('a', 'd', 'h'), ('b', 'e', 'i'), ('c', 'f', 'j')]
tuple(zip(L1, L2, L3)) 
# 输出：(('a', 'd', 'h'), ('b', 'e', 'i'), ('c', 'f', 'j'))

# zip 函数通常使用在循环迭代中
for a, b, c in zip(L1, l2, L3):
    print(a, b, c)

# enumerate是一种特殊的打包，它可以在迭代时绑定迭代元素的遍历序号
L = list('abcd')
for index, value in enumerate(L):
    print(index, value)
# 输出：
'''
0 a
1 b
2 c
3 d
'''

# zip 函数也可以实现这个功能
for index, value in zip(range(len(L)), L)

# zip 函数用于对两个字典建立映射
dict(zip(L1, L2))
```

##### 二、Numpy 基础

###### 1. 数组的构造

```python
# 一般通过 array 构造数组
import numpy as np
np.array([1,2,3])

# 特殊数组
# 等差数组：
np.linspace(1,5,11) # 起始、终止（包含）、样本个数
np.arange(1,5,2) # 起始、终止（不包含）、步长
# 特殊矩阵
np.zeros((2,3)) # 传入元组表示行列数
np.eye(3) # 3*3的单位矩阵
np.eye(3, k=1) # 偏移主对角线1个单位的伪单位矩阵
np.full((2,3), 10) # 元组传入大小，10表示填充数值
np.full((2,3), [1,2,3]) # 每行填入相同的列表
# 随机矩阵
np.random.rand(3) # 生成服从0-1均匀分布的三个随机数
np.random.rand(3, 3) # 注意这里传入的不是元组，每个维度大小分开输入
np.random.randn(3) # 标准正态的随机数组
np.random.randn(2, 2) # 注意这里传入的不是元组，每个维度大小分开输入
np.random.randint(low, high, size) # randint可以指定生成随机整数的最小值最大值（不包含）和维度大小
# choice 可以从给定的列表中，以一定概率和方式抽取结果，当不指定概率时为均匀采样，默认抽取方式为有放回抽样
my_list = ['a', 'b', 'c', 'd']
np.random.choice(my_list, 2, replace=False, p=[0.1, 0.7, 0.1 ,0.1])
# seed 随机种子，它能够固定随机数的输出结果
np.random.seed(0)
np.random.rand()
```

###### 2. np 数组的变形与合并

```python
# 转置 T
np.zeros((2,3)).T

# 合并操作：r_, c_ 分别表示上下合并和左右合并
np.r_[np.zeros((2,3)),np.zeros((2,3))]
# 一维数组和二维数组进行合并时，应当把其视作列向量，在长度匹配的情况下只能够使用左右合并的c_操作：
np.c_[np.array([0,0]),np.zeros((2,3))]

# 维度变换
# reshape能够帮助用户把原数组按照新的维度重新排列。在使用时有两种模式，分别为C模式和F模式，分别以逐行和逐列的顺序进行填充读取
target.reshape((4,2), order='C') # 按照行读取和填充
target.reshape((4,2), order='F') # 按照列读取和填充
```

###### 3. np 数组的切片与索引

```python
# 数组的切片模式支持使用slice类型的start:end:step切片，还可以直接传入列表指定某个维度的索引进行切片
target[:-1, [0,2]]
```

###### 4. 常用函数

```python
# where 条件函数
a = np.array([-1,1,-1,0])
np.where(a>0, a, 5) # 对应位置为True时填充a对应元素，否则填充5
# 返回索引的函数
a = np.array([-2,-5,0,1,3,-1])
np.nonzero(a) # 返回非零数的索引
a.argmax() # 返回最大数的索引
a.argmin() # 返回最小数的索引
# any all 函数
a = np.array([0,1])
a.any() # 指当序列至少存在一个 True或非零元素时返回True，否则返回False
a.all() # all指当序列元素全为 True或非零元素时返回True，否则返回False
# 累计函数和累加函数
a = np.array([1,2,3])
a.cumprod() # 累积函数，输出([1,2,6])
a.cumsum() # 累加函数，输出([1,3,6])
# diff 函数
diff表示和前一个元素做差，由于第一个元素为缺失值，因此在默认参数情况下，返回长度是原数组减1
# 统计函数
max, min, mean, median, std, var, sum, quantile
# nan*类型的函数
对于含有缺失值的数组，它们返回的结果也是缺失值，如果需要略过缺失值，则使用相关的nan*函数
# 协方差和相关系数
cov, corrcoef
# axis 参数
axis=0时结果为列的统计指标，当axis=1时结果为行的统计指标
```

#### 第二章 pandas 基础

##### 一、文件的读取和写入

###### 1. 文件读取

```python
# 基础读取文件
df_csv = pd.read_csv('../data/my_csv')
# 参数 header=None 表示第一行不作为列名
pd.read_table('../data/my_table.txt', header=None)
# index_col 表示把某一列或几列作为索引
pd.read_csv('../data/my_csv.csv', index_col=['col1', 'col2'])
# usecols 表示读取列的集合
pd.read_table('../data/my_table.txt', usecols=['col1', 'col2'])
# parse_dates表示需要转化为时间的列
```

###### 2. 数据写入

```python
# index=False 可以把索引在保存的时候去掉
df_csv.to_csv('../data/my_csv_saved.csv', index=False)
# to_csv可以保存为txt文件，并且允许自定义分隔符；还可以保存为markdown和latex
df_txt.to_csv('../data/my_txt_saved.txt', sep='\t', index=False)
print(df_csv.to_markdown())
print(df_csv.to_latex())
```

##### 二、基本数据结构

###### 1. Series

```python
# 存储一维值，属于object类型
s = pd.Series(data = [100, 'a', {'dic1':5}],
              index = pd.Index(['id1', 20, 'third'], name='my_idx'),
              dtype = 'object',
              name = 'my_name')
s.values
s.index
s.dtypes
s.name
s.shape # 获取序列长度
```

###### 2. DataFrame

```python
# DataFrame在Series的基础上增加了列索引，一个数据框可以由二维的data与行列索引来构造
df = pd.DataFrame(data = {'col_0': [1,2,3],
                          'col_1':list('abc'),
                          'col_2': [1.2, 2.2, 3.2]},
                  index = ['row_%d'%i for i in range(3)])
```

##### 三、常用基本函数

###### 1. 汇总函数

```python
df.head()
df.tail()
df.info() # 返回表的信息概况
df.describe() # 返回表中数值列对应的主要统计量
```

###### 2. 特征统计函数

```python
# 基础函数
sum, mean, median, var, std, max, min
quantile # 分位数
count # 非缺失值个数
idxmax # 最大值对应的索引
# 上面这些所有的函数，由于操作后返回的是标量，所以又称为聚合函数，它们有一个公共参数axis，默认为0代表逐列聚合，如果设置为1则表示逐行聚合
```

###### 3. 唯一值函数

```python
# unique和nunique可以分别得到其唯一值组成的列表和唯一值的个数
df['School'].unique()
# 输出：array(['Shanghai Jiao Tong University', 'Peking University',
       'Fudan University', 'Tsinghua University'], dtype=object)

df['School'].nunique()
# 输出：4

# value_counts可以得到唯一值和其对应出现的频数
df['School'].value_counts()

# drop_duplicates 观察多个列组合的唯一值
df_demo.drop_duplicates(['Gender', 'Transfer'], keep='last')
df_demo.drop_duplicates(['Name', 'Gender'], keep=False).head() # 保留只出现过一次的性别和姓名组合
df['School'].drop_duplicates() # 在Series上也可以使用
# duplicated和drop_duplicates的功能类似，但前者返回了是否为唯一值的布尔列表
```

###### 4. 替换函数

```python
# 在replace中，可以通过字典构造，或者传入两个列表来进行替换
df['Gender'].replace({'Female':0, 'Male':1}).head()
df['Gender'].replace(['Female', 'Male'], [0, 1]).head()

# replace还有一种特殊的方向替换，指定method参数为ffill则为用前面一个最近的未被替换的值进行替换，bfill则使用后面最近的未被替换的值进行替换
s = pd.Series(['a', 1, 'b', 2, 1, 1, 'a'])
s.replace([1, 2], method='ffill')
# 输出：
'''
0    a
1    a
2    b
3    b
4    b
5    b
6    a
'''

# where函数在传入条件为False的对应行进行替换，而mask在传入条件为True的对应行进行替换，当不指定替换值时，替换为缺失值
s.where(s<0, 100)
```

###### 5. 排序函数

```python
# 对升高的值排序，默认 ascending=True
df_demo.sort_values('Height')
df_demo.sort_value('Height', ascending=False) # 降序排列
# 多元素排序问题df_demo.sort_value(['Weight','Height'],ascending=[True,False])
```

###### 6. apply 方法

```python
# apply方法常用于DataFrame的行迭代或者列迭代
df_demo = df[['Height', 'Weight']]
def my_mean(x):
     res = x.mean()
     return res
df_demo.apply(my_mean)

# 利用lambda表达式简化书写
df_demo.apply(lambda x:x.mean())

# mad函数返回的是一个序列中偏离该序列均值的绝对值大小的均值
df_demo.apply(lambda x:(x-x.mean()).abs().mean()) # 和mad一样的效果
df_demo.mad()
```

##### 四、窗口对象

###### 1. 滑窗对象

```python
# 要使用滑窗函数，就必须先要对一个序列使用.rolling得到滑窗对象，
# 其最重要的参数为窗口大小window
'''
s = pd.Series([1,2,3,4,5])
roller = s.rolling(window = 3)
roller
'''
输出：
Rolling [window=3,center=False,axis=0]

roller.mean()
输出：
'''
1    NaN
2    2.0
3    3.0
4    4.0
'''

# 其他滑窗函数
pd.shift(n) # 分别表示取向前第n个元素的值
pd.diff(n) # 与向前第n个元素做差
pd.pcd_change(n) # 与向前第n个元素相比计算增长率
```

###### 2. 扩张窗口

```python
# 设序列为a1, a2, a3, a4，则其每个位置对应的窗口即[a1]、[a1, a2]、[a1, a2, a3]、[a1, a2, a3, a4]
s = pd.Series([1, 3, 6, 10])
s.expanding().mean()
'''
0    1.000000
1    2.000000
2    3.333333
3    5.000000
'''
```

#### 第三章 索引

##### 一、索引器

###### 1. 表的列索引

```python
# DataFrame取出一列或几列
df.['name']
df.name
df.[['gender','name']]
```

###### 2. 序列的行索引

```python
# 取出索引对应的元素
s = pd.Series([1, 2, 3, 4, 5, 6], index=['a', 'b', 'a', 'a', 'a', 'c'])

s['a']
s[['a','b']]
s[['c':'b':-2]] # 从c到b，-2为步长，包含c和b

# 如果前后端点重复出现，则需要先排序
s.sort_index()['a','b']

# 用整数切片，不包含右端点
s[1:-1:2]
```

###### 3. loc 索引器

```python
# 对于行索引，DataFrame有两种索引器，一种是基于元素的loc索引器，另一种是基于位置的iloc索引器
# loc索引器的一般形式是loc[*, *]，其中第一个*代表行的选择，第二个*代表列的选择，如果省略第二个位置写作loc[*]，这个*是指行的筛选
df_demo = df.set_index('Name')
df_demo.head()

# *为单个元素
df_demo.loc['Qiang Sun'] 
df_demo.loc['Qiang Sun', 'School']  # 同时选择行和列

# *为元素列表，取出列表中所有元素值对应的行或列
df_demo.loc[['Qiang Sun','Quan Zhao'], ['School','Gender']]

# *为切片
df_demo.loc['Gaojuan You':'Gaoqiang Qian', 'School':'Gender']

# *为布尔列表
df_demo.loc[df_demo.Weight>70].head()

# 且条件和或条件
df_demo.loc[(条件1)&(条件2)]
df_demo.loc[(条件1)|(条件2)]
```

###### 4. iloc 索引器

```python
# 前两行前两列
df_demo.iloc[[0, 1], [0, 1]] 
# 切片不包含结束端点
df_demo.iloc[1: 4, 2:4] 
# 传入切片为返回值的函数
df_demo.iloc[lambda x: slice(1, 4)] 
# 使用布尔列表的时候要特别注意，不能传入Series而必须传入序列的values，否则会报错。因此，在使用布尔筛选的时候还是应当优先考虑loc的方式
df_demo.iloc[(df_demo.Weight>80).values].head()
```

###### 5. query 方法

```python
# loc一节中的复合条件查询例子可以如下改写
df.query('((School == "Fudan University")&'
         ' (Grade == "Senior")&'
         ' (Weight > 70))|'
         '((School == "Peking University")&'
         ' (Grade != "Senior")&'
         ' (Weight > 80))')
# 所有属于该Series的方法都可以被调用，和正常的函数调用并没有区别
df.query('Weight > Weight.mean()').head()

# NOTE
对于含有空格的列名，需要使用`col name`的方式进行引用
```

###### 6. 随机抽样

```python
# sample函数中的主要参数为n, axis, frac, replace, weights
n: 抽样数量
axis: 抽样的方向（0为行、1为列）
frac: 抽样比例（0.3则为从总体中抽出30%的样本）
replace: 指是否放回,replace=True表示有放回
weights: 每个样本的抽样相对概率     
```

###### 7. 多级索引

#### 第四章 分组

##### 一、分组模式及其对象

###### 1. 分组的一般模式

```python
# df.groupby(分组依据)[数据来源].使用操作
df.groupby('Gender')['Longevity'].mean()
```

###### 2. 分组数据的本质

```python
# 通过一定的复杂逻辑来分组
# 应该先写出分组条件
condition = df.Weight > df.Weight.mean()
# 然后将其传入groupby中
df.groupby(condition)['Height'].mean()
# 传入多列
df.groupby([df['School'], df['Gender']])['Height'].mean()
```

###### 3. Groupy 对象

```python
# 通过ngroups属性，可以得到分组个数
gb.ngroups()

# 通过get_group方法可以直接获取所在组对应的行
gb.get_group(('Fudan University', 'Fresgman'))
```
##### 二、聚合函数

```python
# agg方法能解决的问题
无法同时使用多个函数
无法对特定的列使用特定的聚合函数
无法使用自定义的聚合函数
无法直接对结果的列名在聚合前进行自定义命名

# 使用多个函数
gb.agg(['sum','idmamx','skew'])

# 对特殊的列使用特定的聚合函数，可以通过构造字典传入
gb.agg(['Height':['mean','max'],'weight':'count'])

# 使用自定义函数
gb.agg([lambda x:x.mean()-x.min()])

# 聚合结果重命名
。。。
```
##### 三、变换与过滤
###### 1. 变换函数与 transform 方法

```python
# 最常用的内置变换函数是累计函数：cumcount/cumsum/cumprod/cummax/cummin()/cumsum() 
cumsum() # 依次给出前1、2...n个数的和
cumprod() # 依次给出前1、2...n个数的积

# 自定义变换transform函数

```

######  2. 组索引与过滤

```python
# 过滤在分组中是对于组的过滤，而索引是对于行的过滤，在groupby对象中，定义了filter方法进行组的筛选
gb.filter(lambda x: x.shape[0] > 100).head() # 在原表中通过过滤得到所有容量大于100的组
```

#####  五、跨列分组

######  1. apply 的使用

```python
# apply的自定义函数传入参数与filter完全一致，只不过后者只允许返回布尔值
def BMI(x):
    Height = x['Height']/100
    Weight = x['Weight']
    BMI_value = Weight/Height**2
    return BMI_value.mean()
gb.apply(BMI)
```

####  第五章 变形

#####  一、长宽表的变形

######  1. pivot

```python
# 新生成表的列索引是columns对应列的unique值，而新表的行索引是index对应列的unique值，而values对应了想要展示的数值列
  Class	  Name	    Subject	    Grade
0	1	  San Zhang	Chinese		80
1	1	  San Zhang	Math		75
2	2	  Si Li	    Chinese		90
3	2	  Si Li		Math		85
df.pivot(index='Name', columns='Subject', values='Grade')
# 利用pivot进行变形操作需要满足唯一性的要求，即由于在新表中的行列索引对应了唯一的value，因此原表中的index和columns对应两个列的行组合必须唯一

# pivot相关的三个参数允许被设置为列表，这也意味着会返回多级索引
pivot_multi = df.pivot(index = ['Class', 'Name'],
                       columns = ['Subject','Examination'],
                       values = ['Grade','rank'])
```

######  2. pivot_table

```python
# pivot的使用依赖于唯一性条件，那如果不满足唯一性条件，可以尝试用pivot_table
# pandas中提供了pivot_table来实现，其中的aggfunc参数就是使用的聚合函数
Name	Subject	Grade
0	San Zhang	Chinese	80
1	San Zhang	Chinese	90
2	San Zhang	Math	100
3	San Zhang	Math	90
4	Si Li	Chinese	70
5	Si Li	Chinese	80
6	Si Li	Math	85
7	Si Li	Math	95

df.pivot_table(index = 'Name',
               columns = 'Subject',
               values = 'Grade',
               aggfunc = 'mean')
```

#####  二、索引的变形

######  1. stack 与 unstack

```python
# unstack 把行索引转为列索引
df.unstack(n) # 括号内参数为移动的层号，默认转化最内层
df.unstack([a,b]) # 支持多层转化

```

######  2. 聚合与变形的关系

```python
# 除了带有聚合效果的pivot_table以外，所有的函数在变形前后并不会带来values个数的改变，但由于聚合之后把原来的多个值变为了一个值，因此values的个数产生了变化
```

#####  三、其它变形函数

######  1. crosstab

```

```

######  2. explode

```

```

######  3. get_dummies

```

```

####  第六章 连接

#####  一、关系型连接

######  1. 连接到基本概念

```python
'''
在pandas中的关系型连接函数merge和join中提供了how参数来代表连接形式，分为左连接left、右连接right、内连接inner、外连接outer

只要两边同时出现的值，就以笛卡尔积的方式加入
'''
```

######  2. 值连接

```python
# 两张表根据某一列的值来连接，事实上还可以通过几列值的组合进行连接，这种基于值的连接在pandas中可以由merge函数实现

df1.merge(df2, on='Name', how='left') 
df1.merge(df2, on=['Name', 'Class'], how='left')

# 如果两个表中的列出现了重复的列名，那么可以通过suffixes参数指定
df1 = pd.DataFrame({'Name':['San Zhang'],'Grade':[70]})
df2 = pd.DataFrame({'Name':['San Zhang'],'Grade':[80]})
df1.merge(df2, on='Name', how='left', suffixes=['_Chinese','_Math'])
```

######  3. 索引连接

```python
# pandas中利用join函数来处理索引连接，它的参数选择要少于merge，除了必须的on和how之外，可以对重复的列指定左右后缀lsuffix和rsuffix。其中，on参数指索引名，单层索引时省略参数表示按照当前索引连接
df1 = pd.DataFrame({'Age':[20,30]}, index=pd.Series(['San Zhang','Si Li'],name='Name'))
df2 = pd.DataFrame({'Gender':['F','M']}, index=pd.Series(['Si Li','Wu Wang'],name='Name'))
df1.join(df2, how='left')

df1 = pd.DataFrame({'Grade':[70]}, index=pd.Series(['San Zhang'], name='Name'))
df2 = pd.DataFrame({'Grade':[80]}, index=pd.Series(['San Zhang'], name='Name'))
df1.join(df2, how='left', lsuffix='_Chinese', rsuffix='_Math')
```

#####  二、方向连接

######  1. concat

```python
# concat中，最常用的有三个参数，它们是axis, join, keys，分别表示拼接方向，连接形式，以及在新表中指示来自于哪一张旧表的名字,默认axis=0,join=outer
pd.concat([df1, df2])
pd.concat([df1, df2, df3], 1)
pd.concat([df1, df2], axis=1, join='inner')

# keys参数的使用场景在于多个表合并后，用户仍然想要知道新表中的数据来自于哪个原表，这时可以通过keys参数产生多级索引进行标记，例如，第一个表中都是一班的同学，而第二个表中都是二班的同学，可以使用如下方式合并
pd.concat([df1, df2], keys=['one', 'two'])
```

######  2. 序列与表的合并

```python
# 想要把一个序列追加到表的行末或者列末，则可以分别使用append和assign方法
```

#####  三、类连接操作

######  1. 比较

```python
# 能够比较两个表或者序列的不同处并将其汇总展示
df1.compare(df2)
```

######  2. 组合

```
combine
combine_first
```

####  第七章 缺失数据

#####  一、缺失值的统计和删除

######  1. 缺失信息的统计

```python
# 缺失数据可以使用isna或isnull（两个函数没有区别）来查看每个单元格是否缺失，结合mean可以计算出每列缺失值的比例
df.isnull()
df.isna().mean() 

# 检索出全部为缺失或者至少有一个缺失或者没有缺失的行，可以使用isna, notna和any, all的组合
sub_set = df[['Height', 'Weight', 'Transfer']]
df[sub_set.isna().all(1)] # 全部缺失
df[sub_set.isna().any(1)].head() # 至少有一个缺失
df[sub_set.notna().all(1)].head() # 没有缺失
```

######  2. 缺失信息的删除

```python
# pandas中提供了dropna函数来进行删除操作
# dropna的主要参数为轴方向axis（默认为0，即删除行），删除方式how、删除的非缺失值个数阈值thresh，备选的删除子集subset
res = df.dropna(how = 'any', subset = ['Height', 'Weight'])
res = df.dropna(1, thresh=df.shape[0]-15) # 身高被删除，删除超过15个缺失值的列

# 不用dropna，删除操作的其它方法
res = df.loc[df[['Height','Weight']].notna().all(1)]
res = df.loc[:,~(df.isna().sum()>15)]
```

#####  二、缺失值的填充和插值

######  1. 利用 fillna 进行填充

```python
# fillna中有三个参数是常用的：value, method, limit。其中，value为填充值，可以是标量，也可以是索引到元素的字典映射；method为填充方法，有用前面的元素填充ffill和用后面的元素填充bfill两种类型，limit参数表示连续缺失值的最大填充次数

s.fillna(method='ffill') # 用前面的值向后填充
s.fillna(method='ffill', limit=1) # 连续出现的缺失，最多填充一次
s.fillna(s.mean()) # value为标量
s.fillna({'a': 100, 'd': 200}) # 通过索引映射填充的值

# 有时为了更加合理地填充，需要先进行分组后再操作。例如，根据年级进行身高的均值填充
df.groupby('Grade')['Height'].transform(lambda x: x.fillna(x.mean())).head()
```

######  2. 插值函数

```python

```

#####  三、Nullable 类型

######  1. 缺失记号及其缺陷

```python
# 在python中的缺失值用None表示，该元素除了等于自己本身之外，与其他任何元素不相等
'''
None == False
False
'''
'''
None == []
False
'''
'''
None == ''
False
'''

# numpy中利用np.nan来表示缺失值，该元素除了不和其他任何元素相等之外，和自身的比较结果也返回False
'''
np.nan == np.nan
False
'''

# 在时间序列的对象中，pandas利用pd.NaT来指代缺失值，它的作用和np.nan是一致的
pd.to_timedelta(['30s', np.nan]) 
```

######  2. Nullable 类型的性质

```python
# 
```

######  3. 缺失数据的计算和分组

```python
# 调用函数sum, prod使用加法和乘法的时候，缺失数据等价于被分别视作0和1，即不改变原来的计算结果
# diff, pct_change这两个函数虽然功能相似，但是对于缺失的处理不同，前者凡是参与缺失计算的部分全部设为了缺失值，而后者缺失值位置会被设为 0% 的变化率
# 对于一些函数而言，缺失可以作为一个类别处理，例如在groupby, get_dummies中可以设置相应的参数来进行增加缺失类别
df_nan.groupby('category', dropna=False)['value'].mean()
pd.get_dummies(df_nan.category, dummy_na=True)
```



####  第八章 文本数据

#####  一、str 对象

######  1. str 对象的设计意图

```python
# str对象是定义在Index或Series上的属性，专门用于处理每个元素的文本内容，其内部定义了大量方法，因此对一个序列进行文本处理，首先需要获取其str对象
# 大小写转换
str.upper(var) 
s.str.upper()                     
```

######  2. 索引器

```python
# 对于str对象而言，可理解为其对字符串进行了序列化的操作，例如在一般的字符串中，通过[]可以取出某个位置的元素
s.str[0]
s.str[-1: 0: -2]
```

######  3. string 类型

```python
# 绝大多数对于object和string类型的序列使用str对象方法产生的结果是一致，但是在下面提到的两点上有较大差异：
# 首先，应当尽量保证每一个序列中的值都是字符串的情况下才使用str属性，但这并不是必须的，其必要条件是序列中至少有一个可迭代（Iterable）对象，包括但不限于字符串、字典、列表。对于一个可迭代对象，string类型的str对象和object类型的str对象返回结果可能是不同的


```

#####  二、正则表达式



####  第九章 分类数据

#####  一、cat 对象

######  1. cat 对象的属性

```python

```



####  第十章 时序数据

######  1. Timestamp 的构造与属性

```python
# 单个时间戳可利用pd.Timestamp实现，一般常见的日期格式都能被转换
# 通过year,month,day,hour,min,second可以获得具体数值
```

######  2. Datatime 序列的生成

```python
# 一组时间戳可以组成时间序列，可以用to_datetime和date_range来生成
# 时间戳的格式不满足转换时，可以强制使用format进行匹配
temp = pd.to_datetime(['2020\\1\\1','2020\\1\\3'],format='%Y\\%m\\%d')

# date_range是一种生成连续间隔时间的一种方法，其重要的参数为start, end, freq, periods，它们分别表示开始时间，结束时间，时间间隔，时间戳个数
pd.date_range('2020-1-1','2020-2-28', freq='10D')
```

######  3. dt 对象

```python
# 在时序类型的序列上定义了dt对象来完成许多时间序列的相关操作，可以大致分为三类操作：取出时间相关的属性、判断时间戳是否满足条件、取整操作
# 在这些属性中，经常使用的是dayofweek，它返回了周中的星期情况
s.dt.dayofweek

# 可以通过month_name, day_name返回英文的月名和星期名，注意它们是方法而不是属性
s.dt.month_name()
s.dt.day_name()

# 第二类判断操作主要用于测试是否为月/季/年的第一天或者最后一天
s.dt.is_year_start # 还可选 is_quarter/month_start
s.dt.is_year_end # 还可选 is_quarter/month_end

# 第三类的取整操作包含round, ceil, floor，它们的公共参数为freq，常用的包括H, min, S（小时、分钟、秒）
```

#####  三、时间差

######  1. Timedelta 的生成

```python
# 时间差可以理解为两个时间戳的差
pd.Timedelta(days=1, minutes=25)
pd.Timedelta('1 days 25 minutes')
```

#####  四、日期偏置

######  1. Offset 对象

