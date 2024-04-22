import numpy as np
import librosa

class WhiteNoise:
  def __init__(self):
    pass
    
  def __call__(self, x):
    noised_signal = x + np.random.normal(0, 1, x.shape)
    return librosa.util.normalize(noised_signal)


class RandomVolumeChange:
  def __init__(
      self, 
      high_limit: int=5, 
      low_limit: int=5) -> None:
    
    self.high_limit = high_limit
    self.low_limit = low_limit

  @staticmethod
  def volume_up(x, db):
    return x * RandomVolumeChange._db2float(db)
  
  @staticmethod
  def volume_down(x, db):
    return x * -RandomVolumeChange._db2float(db)
    
  def __call__(self, x):
    db = self._db2float(
        np.random.uniform(-self.low_limit, self.high_limit)
      )
    
    data = None
    if db >= 0:
      data = self.volume_up(x, db)
    else:
      data = self.volume_down(x, db)

    return librosa.util.normalize(data)

  @staticmethod
  def _db2float(db: float):
    return 10 ** (db / 20)
  
  
class TimeStretch:
  def __init__(self, sample_rate=32000):
      self.sample_rate = sample_rate

  def __call__(self, input) -> np.array:
      rate = np.random.uniform(0, 1)
      augmented = librosa.effects.time_stretch(y=input,  rate=rate)
      return librosa.util.normalize(augmented)
  


  

   
    