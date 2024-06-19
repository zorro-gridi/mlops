
class No_SeqDataException(Exception):
    def __init__(self, msg='No_SeqDataException') -> None:
        self.msg = msg

    def __str__(self) -> str:
        return f'\n======> 序列分块数为零！请增大样本数量，或者减少分块序列的长度'
