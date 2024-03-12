from enum import Enum

class FaissIndex(Enum):
    ChunkSize_512_ChunkOverlap_50 = {
        'size': 512,
        'overlap': 50,
    }
    ChunkSize_1500_ChunkOverlap_300 = {
        'size': 1500,
        'overlap': 300,
    }