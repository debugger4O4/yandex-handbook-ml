**Обучение генеративной трансформерной модели с помощью transformers**

В этой работе мы познакомимся на практике с процессом тренировки большой трансформерной языковой модели. Поскольку такая тренировка требует существенных вычислительных ресурсов, выполнять эту работу рекомендуется в Yandex DataSphere, в которой доступны вычислительные узлы с одним или двумя графическими процессорами Tesla V100.

## Архитектура трансформеров

В рамках этой работы мы предполагаем, что вы уже знакомы с архитектурой трансформеров, например, по статье из ML-хэндбука. Также для первоначального знакомства рекомендую заметку Jay Alammar. The Illustrated Transformer, и её частичный русскоязычный перевод.

Мы не будем в рамках работы создавать архитектуру нейросети "с нуля". Если вам интересно изучить реализацию трансформеров — рекомендую посмотреть на NanoGPT. Подробно эта реализация разбирается в этом видео.

## Библиотека transformers и её друзья

Стандартом де-факто в реализации трансформеров служит библиотека transformers от HuggingFace. Она содержит в себе реализацию большого количества используемых трансформерных архитектур, а также ряд полезных инструментов для их обучения. Многие инструменты также оформлены в виде отдельных библиотек, которые хорошо работают вместе:

- **tokenizers**: быстрая реализация различных токенизаторов, позволяющих разделять входной текст на токены.
- **datasets**: манипулирование большими датасетами.
- **evaluate**: вычисление различных метрик и оценка результатов обучения.
- **accelerate**: реализация вычислений на множестве GPU и на вычислительных кластерах.

Для начала установим необходимые библиотеки:

```python
%pip install transformers tokenizers datasets evaluate accelerate
```

## Подготовка датасета

В нашем примере мы будем обучать виртуального Льва Толстого. Для этого возьмем все основные романы писателя и подготовим из них датасет. В качестве отправной точки воспользуемся текстами из библиотеки Мошкова. Соберем ссылки на романы "Анна Каренина", "Война и мир" и др. в один список:

```python
urls = [
    "http://az.lib.ru/t/tolstoj_lew_nikolaewich/text_0039.shtml",
    "http://az.lib.ru/t/tolstoj_lew_nikolaewich/text_0040.shtml",
    "http://az.lib.ru/t/tolstoj_lew_nikolaewich/text_0050.shtml",
    "http://az.lib.ru/t/tolstoj_lew_nikolaewich/text_0060.shtml",
    "http://az.lib.ru/t/tolstoj_lew_nikolaewich/text_0070.shtml",
    "http://az.lib.ru/t/tolstoj_lew_nikolaewich/text_0080.shtml",
    "http://az.lib.ru/t/tolstoj_lew_nikolaewich/text_0090.shtml",
    "http://az.lib.ru/t/tolstoj_lew_nikolaewich/text_1860_dekabristy.shtml",
]
```

Теперь скачаем все материалы и подготовим из них один большой текстовый файл. Для этого удалим HTML-теги и несколько начальных строк в каждом файле.

```python
import html
import re
import requests

def download(url):
    return requests.get(url).text

striptags_re = re.compile(r"(<!--.*?-->|<[^>]*>)")
entity_re = re.compile(r"&([^;]+);")

def to_text(s):
    return html.unescape(striptags_re.sub("", s))

def beautify(s):
    lines = [x.strip() for x in s.split("\n") if x.strip() != ""]
    for i in range(min(100, len(lines))):
        if lines[i] == "-->":
            break
    return "\n".join(lines[i + 1:] if i < 100 else lines)

with open("dataset.txt", "w", encoding="utf-8") as f:
    for u in urls:
        text = beautify(to_text(download(u)))
        f.write(text + "\n\n")
```

## Токенизация

Нейросети работают с числами, поэтому первым этапом является токенизация текста, т.е. разбиение его на атомарные элементы, которые затем можно представить как последовательность индексов в словаре. Мы используем два специальных токена — `[UNK]` для представления неизвестного токена и `[PAD]` для паддинга.

Используя библиотеку `tokenizers`, обучим собственный токенизатор:

```python
import tokenizers as tok
import transformers as tr

tokenizer = tok.Tokenizer(tok.models.BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = tok.pre_tokenizers.Whitespace()
trainer = tok.trainers.BpeTrainer(special_tokens=["[PAD]"])
tokenizer.train(["dataset.txt"], trainer)
tokenizer.enable_padding()
```

## Генеративные трансформеры

Для генерации текста используются архитектуры GPT — Generative Pre-trained Transformers. Мы используем архитектуру GPT-2, которая может неплохо обучиться.

Создаем непосредственно нейросетевую модель GPT2:

```python
config = tr.GPT2Config(
    vocab_size=len(vocab),
    bos_token_id=tokenizer.token_to_id("[CLS]"),
    eos_token_id=tokenizer.token_to_id("[EOS]")
)
gpt = tr.GPT2LMHeadModel(config)
```

Теперь приступим к тренировке модели. Сначала создадим объект `TrainingArguments`, в котором зададим директорию для сохранения промежуточных результатов, число эпох, скорость обучения и т.д.:

```python
targs = tr.TrainingArguments(
    output_dir="gpt2-scratch",
    num_train_epochs=30,
    learning_rate=5e-5,
    warmup_steps=200,
    save_steps=1500
)

trainer = tr.Trainer(
    gpt,
    args=targs,
    train_dataset=dsb["train"],
    tokenizer=ttokenizer,
    data_collator=tr.default_data_collator
)
```

Запустим обучение:

```python
trainer.train()
```

После завершения обучения проверим, как работает генерация текста:

```python
res = gpt.generate(
    **ttokenizer("Пьер закашлялся и", return_tensors="pt").to("cuda"),
    max_new_tokens=150,
    do_sample=True
)
ttokenizer.decode(res[0])
```

## До-обучение GPT-2

Чтобы сократить время обучения, часто используют предварительно обученные модели, которые уже умеют "читать" на нужном языке. Загрузим предобученную модель ruGPT и соответствующий токенизатор:

```python
tokenizer = tr.AutoTokenizer.from_pretrained("ai-forever/rugpt3small_based_on_gpt2")
gpt = tr.GPT2LMHeadModel.from_pretrained("ai-forever/rugpt3small_based_on_gpt2")
```

Перезапустим процесс обучения с новыми параметрами:

```python
targs = tr.TrainingArguments(
    output_dir="gpt2-finetune",
    num_train_epochs=30,
    learning_rate=5e-5,
    warmup_steps=200,
    save_steps=1500
)

trainer = tr.Trainer(
    gpt,
    args=targs,
    train_dataset=dsb["train"],
    tokenizer=tokenizer,
    data_collator=tr.default_data_collator
)

trainer.train()
```

Посмотрим на результат генерации после обучения:

```python
res = gpt.generate(
    **tokenizer("Мне нравится, что вы ", return_tensors="pt").to("cuda"),
    max_new_tokens=50,
    top_k=3,
    do_sample=True
)
tokenizer.decode(res[0])
```

## Параллелизация обучения

Для ускорения процесса обучения обычно используют параллельное обучение на нескольких GPU одновременно. Наиболее распространенным вариантом является параллелизм по данным (Data Parallel Training).

## Заключение

Одной из целей данной работы было показать, что обучение сложных языковых моделей с помощью современных библиотек относительно простая задача, хотя и требующая значительных вычислительных ресурсов. Когда мы выходим за пределы вычислений, которые можно сделать за несколько часов на общедоступных инструментах вроде Google Colab, возникает необходимость в облачных вычислительных ресурсах.

Yandex DataSphere обеспечивает легкий переход от локального Jupyter Notebook или публичного облака Google Colab/Kaggle к выделенной облачной инфраструктуре в Yandex Cloud. В DataSphere вы можете:

- Легко настраивать подключения к облачным хранилищам данных.
- Взаимодействовать с другими участниками проекта.
- Использовать GitHub для контроля версий кода.
- Бережливо расходовать ресурсы благодаря режиму Serverless или возможности легкого переключения между виртуальными вычислителями.

Эффективная работа в DataSphere требует некоторого привыкания, но когда этап адаптации пройден, вы сможете продуктивно использовать этот инструмент и получать удовольствие от работы в нем!