from classes.Similarity import SimHelper
import numpy as np

class Perturber:
    def __init__(self, model: str = 'trocr', max_k: int = 30):
        try:
            self.sim_space = SimHelper.create_sim_space(
                model, 'features/trocr.hdf', num_nearest=max_k)
            self.model = model
            self.max_k = max_k
        except:
            raise Exception(
                f"Could not load similarity space for model {model}. Make sure {model} is a valid model name. Valid model names: imgdot, trocr, detr, beit, clip")

    def perturb_word(self, word: str, k: int = 10, n: float = 0.5):
        if k > self.max_k:
            raise Exception(
                f"Cannot use k={k} for model {self.model}. Maximum k for this model is {self.max_k}.")
        if n > 1 or n < 0:
            raise Exception(
                f"Cannot use n={n} for model {self.model}. n must be between 0 and 1.")

        metadata = {}
        metadata['original'] = word
        metadata['model'] = self.model
        metadata['k'] = k
        metadata['substitutions'] = []

        word = list(word)
        l = len(word)
        idx_to_replace = np.random.choice(
            np.arange(l), size=int(l * n), replace=False)
        for i in idx_to_replace:
            rand_k = np.random.randint(1, k+1)
            neighbor = self.sim_space.topk_neighbors(ord(word[i]), rand_k)[-1]
            neighbor = chr(int(neighbor))
            metadata['substitutions'].append({'from': word[i], 'to': neighbor, 'k': rand_k})
            word[i] = neighbor
            
        perturbation = ''.join(word)
        
        return perturbation, metadata