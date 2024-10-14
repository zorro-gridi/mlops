# coding=utf-8
from setuptools import setup

setup(
    author="Zorro",  ### 作者的名字  也就是你自己的名字
    description="This is a utility, writen by Zorro",  ### 一句话概括一下
    name="mlops",  ### 给你的包取一个名字
    version="1.0",  ### 你的包的版本号
    packages=['mlops'],  ### 这里写的是需要从哪个文件夹下导入python包，目录结构如上面所示，名字必须对上，如果找不到会报错
    install_requires=[  ### 这是你的自定义包中已经导入的一些三方包，这里填上，当自动安装的时候，就会监测本地电脑是否安装，如果没有安装会自动联网按住那个非常方便
    ],
    exclude_package_date={'': ['.gitignore'], '': ['dist'], '': 'build', '': 'utility.egg.info'},
    ### 这是需要排除的文件，也就是只把有用的python文件导入到环境变量中
)
