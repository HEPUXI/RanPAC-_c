import pandas as pd
import argparse
from trainer import train

def main():
    # [Debug-BP0 入口断点]
    # 面试常问: 命令行参数如何流入训练流程。
    # 重点看变量: a.i, a.d
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=int)
    parser.add_argument('-d', type=str)
    a=parser.parse_args()

    # [Debug-BP1 参数装配断点]
    # 重点看变量: exps(参数表), args(当前实验配置字典)
    # 这里把 CSV 的一行配置转换成后续所有模块共享的 args。
    exps=pd.read_csv('./args/'+a.d+'_publish.csv')
    args=exps[exps['ID']==a.i].to_dict('records')[0]
    args['seed']=[args['seed']]
    args['device']=[args['device']]
    args['do_not_save']=False

    # [Debug-BP2 进入训练断点]
    # 从这里进入 trainer.train(args)，后续会实例化 Learner、DataManager。
    train(args)

if __name__ == '__main__':
    main()
