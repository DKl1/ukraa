# import numpy as np
# from typing import List
# from laser_encoders import LaserEncoder as Laser2
# from base_encoder import BaseEncoder
#
#
# class Laser2Encoder(BaseEncoder):
#     def __init__(self):
#         self.encoder = Laser2()
#
#     def embed_sentences(self, sentences: List[str], lang: str) -> np.ndarray:
#         return self.encoder.encode(sentences, lang=lang)
