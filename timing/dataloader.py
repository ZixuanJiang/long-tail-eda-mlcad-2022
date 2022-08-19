import random
import numpy as np
from dgl.distributed import DistDataLoader

class DistDataLoaderSampler(DistDataLoader):
    def __init__(self, collator, batch_size, num_classes, shuffle=False, collate_fn=None, drop_last=False,
            queue_size=None, weighted_sampler=None, drs_flag=False, two_stage_start_epoch=None):
        super(DistDataLoaderSampler, self).__init__(collator.dataset, batch_size, shuffle, collator.collate, drop_last, queue_size)
        self.collator = collator
        self.weighted_sampler = weighted_sampler
        self.num_classes = num_classes
        self.drs_flag = drs_flag
        self.two_stage_start_epoch = two_stage_start_epoch
        self._update_sampler_information()

    def _next_data(self):
        if self.current_pos == len(self.dataset):
            return None
        
        end_pos = 0
        if self.current_pos + self.batch_size > len(self.dataset):
            if self.drop_last:
                return None
            else:
                end_pos = len(self.dataset)
        else:
            end_pos = self.current_pos + self.batch_size

        if (self.weighted_sampler != None) and (not self.drs_flag or (self.drs_flag and self.epoch)):
            idx = set()
            for _ in range(self.current_pos, end_pos):
                if self.weighted_sampler == 'balance':
                    sample_class = random.randint(0, self.num_classes - 1)
                elif self.weighted_sampler == 'square':
                    sample_class = np.random.choice(np.arange(self.num_classes), p=self.square_p)
                elif self.weighted_sampler == 'progressive':
                    sample_class = np.random.choice(np.arange(self.num_classes), p=self.progress_p)
                else:
                    assert False, "Invalid weighted sampler"
                sample_indexes = self.class_dict[sample_class]
                index = random.choice(sample_indexes)
                idx.add(index)
        else:
            idx = self.data_idx[self.current_pos:end_pos].tolist()
        
        ret = [self.dataset[i] for i in idx]
        self.current_pos = end_pos
        return ret

    def _update_sampler_information(self):
        self.num_list = [0] * self.num_classes
        self.class_dict = [[] for _ in range(self.num_classes)]
        for i,idx in enumerate(self.dataset):
            target = self.collator.g.ndata['label'][idx].int().item()
            self.num_list[target] += 1
            self.class_dict[target].append(i)

        num_list_np_array = np.array(self.num_list)
        self.instance_p = num_list_np_array / num_list_np_array.sum()

        max_num = max(self.num_list)
        self.class_balanced_weight = [max_num / i if i != 0 else 0 for i in self.num_list]

        sqrt_num_list = np.sqrt(num_list_np_array)
        self.square_p = sqrt_num_list / sqrt_num_list.sum()
    
    def update_sampler(self, epoch, max_epoch):
        if self.drs_flag:
            self.epoch = max(0, epoch - self.two_stage_start_epoch)
        else:
            self.epoch = epoch
        if self.weighted_sampler == 'progressive':
            self.progress_p = epoch / max_epoch / self.num_classes + (1 - epoch / max_epoch) * self.instance_p

