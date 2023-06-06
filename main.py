import os
import time
import torch
import argparse

from model import SASRec
from utils import *

from torch_ema import ExponentialMovingAverage

def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Beauty', type=str)
parser.add_argument('--save_dir', default='tmp_out')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=901, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--eval_frequency', default=1, type=int)
parser.add_argument('--gpu', default="1", type=str)
parser.add_argument('--alpha', default=0.7, type=float)
parser.add_argument('--teaching_epoch', default=50, type=int)
parser.add_argument('--onlyP', default=False, type=str2bool)
parser.add_argument('--decay', default=0.999, type=float)
parser.add_argument('--recontinue', default=True, type=str2bool)
parser.add_argument('--expand', default=9999, type=int)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if not os.path.isdir(args.save_dir):
    os.makedirs(args.save_dir)
with open(os.path.join(args.save_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True



if __name__ == '__main__':
    # global dataset
    setup_seed(20)
    record = {}
    dataset = data_partition(args.dataset)
    evaluate_pos = {'@1':1, '@5':5, '@10':10, '@20':20, '@50':50}
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    print(num_batch)
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    print('users_num:', usernum)
    print('item_num:', itemnum)
    
    f = open(os.path.join(args.save_dir, 'log.txt'), 'a')
    
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=1, expand = args.expand)
    model = SASRec(usernum, itemnum, args).to(args.device) # no ReLU activation in original SASRec implementation?
    teacher_model = SASRec(usernum, itemnum, args).to(args.device)
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers
    
    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)

    
    
    model.train() # enable model training
    
    

    ema = None
            
    
    # if args.inference_only:
    #     model.eval()
    #     test_ndcg, test_hit = evaluate(model, dataset, args, evaluate_pos)
    #     print('test_ndcg:',test_ndcg)
    #     print('test_hit:',test_hit)
    
    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    
    bce_criterion = torch.nn.BCELoss()

    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    
    valid_best = {'@1':0, '@5':0, '@10':0, '@20':0, '@50':0}
    test_best = {'@1':0, '@5':0, '@10':0, '@20':0, '@50':0}

    folder = args.save_dir

    epoch_start_idx = 1

    if args.recontinue and os.path.exists(os.path.join(folder, "newest_model.pth")):
        model.load_state_dict(torch.load(os.path.join(folder, "newest_model.pth"), map_location=torch.device(args.device)))
        dic = np.load(os.path.join(folder, "dic.npy"), allow_pickle=True).item()
        record = np.load(os.path.join(folder, "record.npy"), allow_pickle=True).item()
        epoch_start_idx = dic["epoch"] + 1
        if "decay" in dic:
            ema = ExponentialMovingAverage(model.parameters(), decay=args.decay)
            ema.to(device = args.device)
            ema.load_state_dict(dic)

    
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        record[epoch] = {'loss': 0}
        if args.inference_only: break # just to decrease identition

        if epoch == args.teaching_epoch - 1:
            ema = ExponentialMovingAverage(model.parameters(), decay=args.decay)
            ema.to(device = args.device)

        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg, pos_exp = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg, pos_exp = np.array(u), np.array(seq), np.array(pos), np.array(neg), np.array(pos_exp)
            # print(seq.shape, neg.shape)
            # exit()
            
            
            if epoch >= args.teaching_epoch: #是否扩展正样本
                pos = pos_exp 

            pos_pred, neg_pred = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_pred.shape, device=args.device), torch.zeros(neg_pred.shape, device=args.device)
            # print("\neye ball check raw_pred:"); print(pos_pred); print(neg_pred) # check pos_pred > 0, neg_pred < 0
            
            if epoch >= args.teaching_epoch:
                # teacher_model.load_state_dict(torch.load(args.save_dir + '/best_valid_model.pth'))
                # teacher_model.eval()
                # tpos_pred, tneg_pred = teacher_model(u, seq, pos, neg)

                with ema.average_parameters():
                    tpos_pred, tneg_pred = model(u, seq, pos, neg)
                    tpos_pred = tpos_pred.detach()
                    tneg_pred = tneg_pred.detach()
                    
                
                pos_labels = pos_labels * (1 - args.alpha) + args.alpha * tpos_pred

                if not args.onlyP: #是否对负样本进行约束
                    neg_labels = args.alpha * tneg_pred

            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            
            loss = bce_criterion(pos_pred[indices], pos_labels[indices])
            loss += bce_criterion(neg_pred[indices], neg_labels[indices])            

            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            record[epoch]['loss'] += loss.item()
            if step == 0:
                print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs

            if epoch >= args.teaching_epoch:
                ema.update()

        if epoch % args.eval_frequency == 0:
            model.eval()
           
            
            print('Evaluating', end='')
            
            if epoch >= args.teaching_epoch:
                with ema.average_parameters():
                    test_ndcg, test_hit = evaluate(model, dataset, args, evaluate_pos)
                    valid_ndcg, valid_hit = evaluate_valid(model, dataset, args, evaluate_pos)
            else:
                test_ndcg, test_hit = evaluate(model, dataset, args, evaluate_pos)
                valid_ndcg, valid_hit = evaluate_valid(model, dataset, args, evaluate_pos)

            print('epoch:%d',epoch)
            print('valid_ndcg:',valid_ndcg)
            print('valid_hit:',valid_hit)
            print('test_ndcg:',test_ndcg)
            print('test_hit:',test_hit)

            f.write('epoch:' + str(epoch) + '\n')
            f.write('valid_ndcg:' + str(valid_ndcg) + '\n')
            f.write('valid_hit:' + str(valid_hit) + '\n')
            f.write('test_ndcg:' + str(test_ndcg) + '\n')
            f.write('test_hit:' + str(test_hit) + '\n')
            f.flush()

            record[epoch]['valid_ndcg'] = valid_ndcg
            record[epoch]['test_ndcg'] = test_ndcg
            record[epoch]['valid_hit'] = valid_hit
            record[epoch]['test_hit'] = test_hit

            
            model.train()
    
            # if valid_ndcg['@10'] > valid_best['@10']:
            #     valid_best = valid_ndcg
            #     folder = args.save_dir
            #     fname = 'best_valid_model.pth'
            #     torch.save(model.state_dict(), os.path.join(folder, fname))

            if test_ndcg['@10'] > test_best['@10']:
                test_best = test_ndcg
                folder = args.save_dir
                fname = 'best_test_model.pth'
                if epoch >= args.teaching_epoch:
                    with ema.average_parameters():
                        torch.save(model.state_dict(), os.path.join(folder, fname))
                else:
                    torch.save(model.state_dict(), os.path.join(folder, fname))

        # save!!!
        torch.save(model.state_dict(), os.path.join(folder, "newest_model.pth"))
        if epoch >= args.teaching_epoch:
            dic = ema.state_dict()
        else:
            dic = {}
        dic["epoch"] = epoch
        dic["valid_best"] = valid_best
        dic["test_best"] = test_best
        np.save(os.path.join(folder, "dic"), dic)
        np.save(args.save_dir + '/record', record)
    
    f.close()
    sampler.close()
    print('best:', test_best)
    
    print("Done")

