import collections
from functools import partial as bind
from typing import Callable, Any
import elements
import embodied
import numpy as np
from embodied.core import base

def train(
    make_agent: Callable[[], embodied.Agent],
    make_replay: Callable[[], embodied.Replay],
    make_env: Callable[[], embodied.Env],
    make_stream: Callable[[], base.Stream],
    make_logger: Callable[[], elements.Logger],
    args: Any,
):
    """
    Train a Dreamer agent on an environment.
    """
    # create agent, replay buffer, and logger
    agent = make_agent()
    replay = make_replay()
    logger = make_logger()

    # set up logging directory and tracking utilities
    logdir = elements.Path(args.logdir)
    step = logger.step
    usage = elements.Usage(**args.usage)
    train_agg = elements.Agg()
    epstats = elements.Agg()
    episodes = collections.defaultdict(elements.Agg)
    policy_fps = elements.FPS()
    train_fps = elements.FPS()

    # compute how often to train based on ratio and batch size
    batch_steps = args.batch_size * args.batch_length
    should_train = elements.when.Ratio(args.train_ratio / batch_steps)
    should_log = embodied.LocalClock(args.log_every)
    should_report = embodied.LocalClock(args.report_every)
    should_save = embodied.LocalClock(args.save_every)

    # define logging function for each transition
    @elements.timer.section('logfn')
    def logfn(tran, worker):
        episode = episodes[worker]
        tran['is_first'] and episode.reset()

        # accumulate basic metrics
        episode.add('score', tran['reward'], agg='sum')
        episode.add('length', 1, agg='sum')
        episode.add('rewards', tran['reward'], agg='stack')

        # accumulate additional image and logging metrics
        for key, value in tran.items():
            if value.dtype == np.uint8 and value.ndim == 3:
                if worker == 0:
                    episode.add(f'policy_{key}', value, agg='stack')
            elif key.startswith('log/'):
                assert value.ndim == 0, (key, value.shape, value.dtype)
                episode.add(key + '/avg', value, agg='avg')
                episode.add(key + '/max', value, agg='max')
                episode.add(key + '/sum', value, agg='sum')

        # process and log episode summary when finished
        if tran['is_last']:
            result = episode.result()
            logger.add({
                'score': result.pop('score'),
                'length': result.pop('length'),
            }, prefix='episode')
            rew = result.pop('rewards')
            if len(rew) > 1:
                result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
            epstats.add(result)

    # create environment functions and initialize driver
    fns = [bind(make_env, i) for i in range(args.envs)]
    driver = embodied.Driver(fns, parallel=not args.debug)

    # register callbacks for each step in environment
    driver.on_step(lambda tran, _: step.increment())
    driver.on_step(lambda tran, _: policy_fps.step())
    driver.on_step(replay.add)
    driver.on_step(logfn)

    # prepare data streams for training and reporting
    stream_train = iter(agent.stream(make_stream(replay, 'train')))
    stream_report = iter(agent.stream(make_stream(replay, 'report')))

    carry_train = [agent.init_train(args.batch_size)]
    carry_report = agent.init_report(args.batch_size)

    # define training function executed during stepping
    def trainfn(tran, worker):
        if len(replay) < args.batch_size * args.batch_length:
            return
        for _ in range(should_train(step)):
            with elements.timer.section('stream_next'):
                batch = next(stream_train)
            carry_train[0], outs, mets = agent.train(carry_train[0], batch)
            train_fps.step(batch_steps)
            if 'replay' in outs:
                replay.update(outs['replay'])
            train_agg.add(mets, prefix='train')

    driver.on_step(trainfn)

    # set up checkpointing system
    cp = elements.Checkpoint(logdir / 'ckpt')
    cp.step = step
    cp.agent = agent
    cp.replay = replay
    if args.from_checkpoint:
        elements.checkpoint.load(args.from_checkpoint, dict(
            agent=bind(agent.load, regex=args.from_checkpoint_regex)))
    cp.load_or_save()

    # start training loop
    print('Start training loop')
    policy = lambda *args: agent.policy(*args, mode='train')
    driver.reset(agent.init_policy)
    while step < args.steps:
        driver(policy, steps=10)
        # report periodically based on replay content
        if should_report(step) and len(replay):
            agg = elements.Agg()
            for _ in range(args.consec_report * args.report_batches):
                carry_report, mets = agent.report(carry_report, next(stream_report))
                agg.add(mets)
            logger.add(agg.result(), prefix='report')

        # log metrics if it's time
        if should_log(step):
            logger.add(train_agg.result())
            logger.add(epstats.result(), prefix='epstats')
            logger.add(replay.stats(), prefix='replay')
            logger.add(usage.stats(), prefix='usage')
            logger.add({'fps/policy': policy_fps.result()})
            logger.add({'fps/train': train_fps.result()})
            logger.add({'timer': elements.timer.stats()['summary']})
            logger.write()

        # save model state if needed
        if should_save(step):
            cp.save()

    # finalize logging
    logger.close()
