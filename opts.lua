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
    cmd:option('-att_hid_size', 512, 'the hidden size of the attention MLP; 0 if not using hidden layer')
    
    cmd:option('-val_size', 4000, 'Validation set size')
    cmd:option('-test_size', 4000, 'Test set size')

    cmd:option('-use_attention', true, 'Use attention or not')
    cmd:option('-use_noun', true, 'Use noun or not')
    cmd:option('-reason_weight', 1.0, 'weight of reasoning loss')

    -- cmd:option('-use_reasoning', true, 'Use reasoning. Will use attention in default.')
    cmd:option('-model_pack', 'reason_att', 'the model package to use, can be reason_att, reasoning, or soft_att_lstm')
    cmd:option('-reason_step', 5, 'Reasoning steps before the decoder')

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

    cmd:option('-id2noun_file', 'data/annotations/id2nouns.txt', 'Path to the id 2 nouns file')

    ------------ Training options --------------------
    cmd:option('-nEpochs', 100, 'Number of epochs in training')
    -- cmd:option('-eval_period', 12000, 'Every certain period, evaluate current model')
    -- cmd:option('-loss_period', 2400, 'Every given number of iterations, compute the loss on train and test')
    cmd:option('-batch_size', 32, 'Batch size in SGD')
    cmd:option('-val_batch_size', 10, 'Batch size for testing')
    cmd:option('-LR', 0.01, 'Initial learning rate')
    cmd:option('-truncate', 30, 'Text longer than this size gets truncated. -1 for no truncation.')
    cmd:option('-max_eval_batch', 50, 'max number of instances when calling comp error. 20000 = 4000 * 5')
    cmd:option('-save_file', true, 'whether save model file?')
    cmd:option('-save_file_name', 'attention.1024.model', 'file name for saving model')
    
    ------------ Evaluation options --------------------
    cmd:option('-model', 'models/concat.1024.512.model', 'Model to evaluate')
    cmd:option('-eval_algo', 'beam', 'Evaluation algorithm, beam or greedy')
    cmd:option('-beam_size', 5, 'Beam size in beam search')
    cmd:option('-val_max_len', 20, 'Max length in validation state')
    
    local opt = cmd:parse(arg or {})
    opt.eval_period = math.floor(3000 * 32 / opt.batch_size)
    opt.loss_period = math.floor(600 * 32 / opt.batch_size)
    return opt
end

return M





