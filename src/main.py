from candidate_sampler import CandidateSampler

if __name__ == '__main__':

    sampler = CandidateSampler(config_path='src/config.yaml')
    sampler.prepare_indices()
    sampler.set_thresholds()
    sampler.sample()