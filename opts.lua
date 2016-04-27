local M = {}

function M.parse(arg)    
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Torch-7 Image Captioning')
    cmd:text()
    cmd:text('Options:')

    ------------ Model options ----------------------
    cmd:option('-emb_size', 100, 'Word embedding size')
    cmd:option('-lstm_size', 1024, 'LSTM size')
    cmd:option('-att_size', 196, 'how many attention areas')
    cmd:option('-feat_size', 512, 'the dimension of each attention area')
    cmd:option('-fc7_size', 4096, 'the dimension of fc7')
    
    cmd:option('-val_size', 4000, 'Validation set size')
    cmd:option('-test_size', 4000, 'Test set size')

    cmd:option('-use_attention', true, 'Use attention or not')

    ------------ General options --------------------
    cmd:option('-data', 'data/', 'Path to dataset')
    cmd:option('-train_feat', 'train2014_features_vgg_vd19_conv5', 'Path to pre-extracted training image feature')
    cmd:option('-val_feat', 'val2014_features_vgg_vd19_conv5', 'Path to pre-extracted validation image feature')
    cmd:option('-train_fc7', 'train2014_features_vgg_vd19_fc7', 'Path to pre-extracted training fully connected 7')
    cmd:option('-val_fc7', 'val2014_features_vgg_vd19_fc7', 'Path to pre-extracted validation fully connected 7')
    cmd:option('-train_anno', 'annotations/captions_train2014.json', 'Path to training image annotaion file')
    cmd:option('-val_anno', 'annotations/captions_val2014.json', 'Path to validation image annotaion file')
    cmd:option('-nGPU', 1, 'Index of GPU to use, 0 means CPU')
    cmd:option('-seed', 123, 'Random number seed')

    ------------ Training options --------------------
    cmd:option('-nEpochs', 100, 'Number of epochs in training')
    -- cmd:option('-eval_period', 12000, 'Every certain period, evaluate current model')
    -- cmd:option('-loss_period', 2400, 'Every given number of iterations, compute the loss on train and test')
    cmd:option('-batch_size', 32, 'Batch size in SGD')
    cmd:option('-val_batch_size', 10, 'Batch size for testing')
    cmd:option('-LR', 0.1, 'Initial learning rate')
    cmd:option('-truncate', 30, 'Text longer than this size gets truncated. -1 for no truncation.')
    cmd:option('-max_eval_batch', 50, 'max number of instances when calling comp error. 20000 = 4000 * 5')
    cmd:option('-save_file', true, 'whether save model file?')
    cmd:option('-save_file_name', 'attention.1024.model', 'file name for saving model')

    local opt = cmd:parse(arg or {})
    opt.eval_period = math.floor(3000 * 32 / opt.batch_size)
    opt.loss_period = math.floor(600 * 32 / opt.batch_size)
    return opt
end

return M





