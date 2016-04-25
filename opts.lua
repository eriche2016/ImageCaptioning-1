local M = {}

function M.parse(arg)    
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Torch-7 Image Captioning')
    cmd:text()
    cmd:text('Options:')

    ------------ Model options ----------------------
    cmd:option('-emb_size', 100, 'Word embedding size')
    cmd:option('-lstm_size', 4096, 'LSTM size')
    cmd:option('-att_size', 196, 'Test set size')
    cmd:option('-feat_size', 512, 'Test set size')
    
    cmd:option('-val_size', 5000, 'Validation set size')
    cmd:option('-test_size', 5000, 'Test set size')

    ------------ General options --------------------
    cmd:option('-data', '/usr0/home/yuexinw/research/701/data', 'Path to dataset')
    cmd:option('-train_feat', 'train2014_features_vgg_vd19_conv5', 'Path to pre-extracted training image feature')
    cmd:option('-val_feat', 'val2014_features_vgg_vd19_conv5', 'Path to pre-extracted validation image feature')
    cmd:option('-train_anno', 'annotations/captions_train2014.json', 'Path to training image annotaion file')
    cmd:option('-val_anno', 'annotations/captions_val2014.json', 'Path to validation image annotaion file')
    cmd:option('-nGPU', 4, 'Index of GPU to use, 0 means CPU')
    cmd:option('-seed', 123, 'Random number seed')

    ------------ Training options --------------------
    cmd:option('-nEpochs', 100, 'Number of epochs in training')
    cmd:option('-eval_period', 100, 'Every certain period, evaluate current model')
    cmd:option('-batch_size', 32, 'Batch size in SGD')
    cmd:option('-LR', 0.01, 'Initial learning rate')

    local opt = cmd:parse(arg or {})
    return opt
end

return M





