from openai import OpenAI
from pathlib import Path
import os

home_dir = Path(os.path.expanduser('~'))
api_key = Path(home_dir / 'project/pycharm/ChatGpt_API_Key.txt').open().read()


client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=api_key,
    base_url="https://api.chatanywhere.tech/v1"
    )


# 非流式响应
def gpt_35_api(messages: list):
    """
    Desc:
        为提供的对话消息创建新的回答
    Args:
        messages (list): 完整的对话消息
    """
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
        )
    print(completion.choices[0].message.content)


def gpt_35_api_stream(messages: list):
    """
    Desc:
        为提供的对话消息创建新的回答 (流式传输)
    Args:
        messages (list): 完整的对话消息
    """
    stream = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        if len(chunk.choices) > 0:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")



if __name__ == '__main__':
    content = '''
        这是一个文本分类任务，现在给你几个案例：AI热潮涌向影视行业 AI电影的时代来了吗	AI#传媒#TMT
        传媒接棒成AI主阵地！如何把握低位个股异动机会？	AI#传媒#TMT
        AI孙燕姿出道即成顶流 人工智能席卷音乐圈	AI#传媒#TMT。
        “AI+”堪比13年的“移动互联网+”	AI
        AIGC真正的分歧产生了！年内市值暴增4000亿，还值得“爱”吗？但斌等大佬最新发声“表态”，李开复也有新观点	AI
        微软把聊天机器人技术植入输入法 微软输入法整合ChatGPT	ChatGPT#机器人
        2023年“清明档”国内电影预售总票房突破2000万元	传媒#TMT
        4月全国电影市场总票房已破10亿元	传媒#TMT
        爆发！传媒股强势攀升，行业“奇点”将至？	传媒#TMT

        需要注意的问题：这是一个对文本进行多标签分类的任务，如果根据语义理解，某个标签与文本内容不符，则不应该将它归类到某一个标签。
        即，一个文本可以属于多个标签，也可以只有一个。不要拘泥于例子的示范。
        现在请问：
            GPT革命｜高盛：全球18%工作将由生成式AI自动化完成
            GPT革命｜智谱AI张鹏：500一600亿参数是大模型的门槛
            OpenAI发布ChatGPT安全方法 ChatGPT被禁止使用
            阿里版GPT官宣内测 重磅！阿里云大模型来了！为啥叫 “通义千问”？     阿里版GPT为啥叫通义千问
            阿里大模型通义千问开启企业邀测 阿里版GPT通义千问来了
            阿里云CTO周靖人：阿里云将开放“通义千问” 助力企业打造专属大模型
            GPT革命｜高盛：全球18%工作将由生成式AI自动化完成
            GPT革命｜智谱AI张鹏：500一600亿参数是大模型的门槛
            OpenAI发布ChatGPT安全方法 ChatGPT被禁止使用
            阿里版GPT官宣内测 重磅！阿里云大模型来了！为啥叫 “通义千问”？     阿里版GPT为啥叫通义千问
            阿里大模型通义千问开启企业邀测 阿里版GPT通义千问来了
            阿里云CTO周靖人：阿里云将开放“通义千问” 助力企业打造专属大模型
            GPT革命｜高盛：全球18%工作将由生成式AI自动化完成
            GPT革命｜智谱AI张鹏：500一600亿参数是大模型的门槛
            OpenAI发布ChatGPT安全方法 ChatGPT被禁止使用
            阿里版GPT官宣内测 重磅！阿里云大模型来了！为啥叫 “通义千问”？     阿里版GPT为啥叫通义千问
            阿里大模型通义千问开启企业邀测 阿里版GPT通义千问来了
            阿里云CTO周靖人：阿里云将开放“通义千问” 助力企业打造专属大模型
            GPT革命｜高盛：全球18%工作将由生成式AI自动化完成
            GPT革命｜智谱AI张鹏：500一600亿参数是大模型的门槛
            OpenAI发布ChatGPT安全方法 ChatGPT被禁止使用
            阿里版GPT官宣内测 重磅！阿里云大模型来了！为啥叫 “通义千问”？     阿里版GPT为啥叫通义千问
            阿里大模型通义千问开启企业邀测 阿里版GPT通义千问来了
            阿里云CTO周靖人：阿里云将开放“通义千问” 助力企业打造专属大模型
        以上文本可分别归于哪些标签？分别给出对应的标签结果，你预测的标签需要严格在案例提供的标签范围内！
    '''
    messages = [
        {'role': 'user', 'content': content},
        ]
    # 非流式调用
    # gpt_35_api(messages)
    # 流式调用
    gpt_35_api_stream(messages)

    # import time
    # for _ in range(100):
    #     gpt_35_api_stream(messages)
    #     time.sleep(0.5)