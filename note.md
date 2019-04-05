# Bert 源代码分析

## tokenization.py

对数据集的预处理

### validate_case_matches_checkpoint
检验配置是否与**checkpoint**一致

### convert_to_unicode
将文本转换为unicode格式，假设utf-8格式

### printable_text
将文本进行编码以方便控制台或者日志输出

### load_vocab
载入词表文件以获取词表字典 key: token, value: index

### convert_by_vocab
利用**vocab/in_vocab**，将**token/id**序列转换为对应的**id/index**序列

### convert_tokens_to_ids / convert_ids_to_tokens
调用convert_by_vocab函数

### whitespace_tokenize
删除文本的空格，并按照文本中的空格进行**split**操作，返回列表

### class FullTokenizer
*  vocab 词表；
*  in_vocab 逆词表（通过遍历vocab，并交换键值对）；
*  basic_tokenizer BasicTokenizer对象；
*  wordpiece_tokenizer WordpieceTokenizer对象。

#### tokenize

### class BasicTokenizer
* do_lower_case 是否小写

#### tokenize
先调用**convert_to_unicode**将文本转换为unicode格式，再调用**_clean_text**对文本去除无效字符等操作，然后调用**_token_chinese_chars**和
**whitespace_tokenize**对文本进行中文和空格处理，接着迭代遍历，并根据**do_lower_case**判断是否对文本进行小写处理，而且每次迭代中都会调用**_run_split_on_punc**切分文本，最后在进行空格处理

#### _run_strip_accents
从文本中删除重音符号。先将文本转换为NFD（表示字符应该分解为多个组合字符表示）格式，迭代遍历，Mn（是非间距字符，这指示基字符的修改。）跳过，其他正常处理。

#### _run_split_on_punc
根据标点符号切分文本

#### _tokenize_chinese_chars
迭代遍历，对中文字符前后添加空格，其他不变

#### _is_chinese_char
通过字符编码判断是否为中文字符

#### _clean_text
迭代遍历，对无效字符和进行处理，控制字符，字符unicode编码为 0 或 0xFFFD的跳过，其他正常

### WordpieceTokenizer

### _is_control
判断是否为控制字符，"Cc"（字符是控制代码）和"Cf"(格式字符，通常不显示，但影响文本布局和文本处理)类别，但是不处理"\t"、"\n"、"\r"