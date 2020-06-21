from __future__ import print_function

import os
import sys
import datetime
import torch

class BaseSaver:
    def __init__(self, path):
        self.path = path
        os.system('mkdir -p %s'%path)
        self.model_lst = self._getList()

    def _getList(self):
        lst = [x for x in sorted(os.listdir(self.path)) if x.endswith('.pkl')]
        return lst

    def LoadLatestModel(self, model, model_name=None):
        step = 0
        if len(self.model_lst) == 0:
            print("Empty model folder! Using initial weights")
            #model.init_weights()
            return None, step
        
        # No providing model_name, use latest one
        if model_name is not None:
            model_name = [x for x in self.model_lst if '_%s.'%(model_name) in x][0]
        else:
            model_name = self.model_lst[-1]
        
        print('Load: %s'%model_name)
        name = self.path + '/' + model_name
        params = torch.load(name)
        model.load_state_dict(params, strict=False)
        
        # Use step if it's detected
        strs = model_name.replace('.pkl', '').split('_')
        if len(strs) == 3:
            step = int(strs[-1]) + 1
        
        return name, step

    def Save(self, model, step):
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S_{}'.format(step))

        name = '%s/model_%s.pkl'%(self.path, now)

        torch.save(model.state_dict(), name)


if __name__ == '__main__':
    print(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
