class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'COD10K':
            return './dataset/COD10K/'
        elif dataset == 'MoCA':
            return '/Users/mac/data/cv/MoCA_Video/'
        elif dataset == 'CAD2016':
            return './dataset/CAD2016/CamouflagedAnimalDataset/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
