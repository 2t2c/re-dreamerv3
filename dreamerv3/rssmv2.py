import math
import einops
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

f32 = jnp.float32
sg = jax.lax.stop_gradient


# TRSSM = Transformer Recurrent State Space Model
class RSSM(nj.Module):
    # hyperparameters with defaults
    deter: int = 4096  # size of deterministic hidden state
    hidden: int = 2048  # hidden layer size for networks
    stoch: int = 32  # number of stochastic variables per class
    classes: int = 32  # number of discrete classes per stochastic variable
    norm: str = 'rms'  # normalization type
    act: str = 'gelu'  # activation function
    unroll: bool = False  # whether to use scanning for unrolling time
    unimix: float = 0.01  # uniform mixing for discrete distributions
    adaptive_unimix: bool = True  # whether to adaptively mix unimix
    outscale: float = 1.0  # output scale for logits
    imglayers: int = 2  # layers in image decoder (prior)
    obslayers: int = 1  # layers in observation encoder (posterior)
    dynlayers: int = 1  # number of layers in dynamics model
    absolute: bool = False  # whether to use only tokens or concatenate with deter
    blocks: int = 8  # number of blocks for BlockLinear layers
    free_nats: float = 1.0  # threshold for KL regularization
    trf_layers: int = 4 # transformer blocks
    attention_heads: int = 8 # attention heads
    gating: bool = False  # whether to use gating in transformer

    def __init__(self, act_space, **kw):
        # ensure compatibility with BlockLinear
        assert self.deter % self.blocks == 0
        # action space description
        self.act_space = act_space
        # additional kwargs passed to submodules
        self.kw = kw

    @property
    def entry_space(self):
        # Shape of latent entries returned at each timestep
        return dict(
            deter=elements.Space(np.float32, self.deter),
            stoch=elements.Space(np.float32, (self.stoch, self.classes)))

    def initial(self, bsize):
        # initial state (zeroed)
        carry = nn.cast(dict(
            deter=jnp.zeros([bsize, self.deter], f32),
            stoch=jnp.zeros([bsize, self.stoch, self.classes], f32)))
        return carry

    def truncate(self, entries, carry=None):
        # extract last timestep to serve as initial state for next episode
        assert entries['deter'].ndim == 3, entries['deter'].shape
        carry = jax.tree.map(lambda x: x[:, -1], entries)
        return carry

    def starts(self, entries, carry, nlast):
        # prepare sequences starting from last n steps
        B = len(jax.tree.leaves(carry)[0])
        return jax.tree.map(
            lambda x: x[:, -nlast:].reshape((B * nlast, *x.shape[2:])), entries)

    def observe(self, carry, tokens, action, reset, training, single=False):
        # encodes observations (posterior) given inputs
        carry, tokens, action = nn.cast((carry, tokens, action))
        if single:
            carry, (entry, feat) = self._observe(carry, tokens, action, reset, training)
            return carry, entry, feat
        else:
            unroll = jax.tree.leaves(tokens)[0].shape[1] if self.unroll else 1
            carry, (entries, feat) = nj.scan(
                lambda carry, inputs: self._observe(carry, *inputs, training),
                carry, (tokens, action, reset), unroll=unroll, axis=1)
            return carry, entries, feat

    def _observe(self, carry, tokens, action, reset, training):
        # processes one step of observation encoding
        deter, stoch, action = nn.mask((carry['deter'], carry['stoch'], action), ~reset)
        action = nn.DictConcat(self.act_space, 1)(action)
        action = nn.mask(action, ~reset)
        # transformer core
        deter = self.trf_core(deter, stoch, action)
        tokens = tokens.reshape((*deter.shape[:-1], -1))
        x = tokens if self.absolute else jnp.concatenate([deter, tokens], -1)
        for i in range(self.obslayers):
            x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
            x = nn.act(self.act)(self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))
        logit = self._logit('obslogit', x)
        stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
        carry = dict(deter=deter, stoch=stoch)
        feat = dict(deter=deter, stoch=stoch, logit=logit)
        entry = dict(deter=deter, stoch=stoch)
        assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
        return carry, (entry, feat)

    def imagine(self, carry, policy, length, training, single=False):
        # predicts future latent states using learned dynamics and a policy
        if single:
            action = policy(sg(carry)) if callable(policy) else policy
            actemb = nn.DictConcat(self.act_space, 1)(action)
            deter = self.trf_core(carry['deter'], carry['stoch'], actemb)
            logit = self._prior(deter)
            stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
            carry = nn.cast(dict(deter=deter, stoch=stoch))
            feat = nn.cast(dict(deter=deter, stoch=stoch, logit=logit))
            assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
            return carry, (feat, action)
        else:
            unroll = length if self.unroll else 1
            if callable(policy):
                carry, (feat, action) = nj.scan(
                    lambda c, _: self.imagine(c, policy, 1, training, single=True),
                    nn.cast(carry), (), length, unroll=unroll, axis=1)
            else:
                carry, (feat, action) = nj.scan(
                    lambda c, a: self.imagine(c, a, 1, training, single=True),
                    nn.cast(carry), nn.cast(policy), length, unroll=unroll, axis=1)
            # We can also return all carry entries but it might be expensive.
            # entries = dict(deter=feat['deter'], stoch=feat['stoch'])
            # return carry, entries, feat, action
            return carry, feat, action

    def loss(self, carry, tokens, acts, reset, training):
        # computes KL divergence loss between posterior and prior
        metrics = {}
        carry, entries, feat = self.observe(carry, tokens, acts, reset, training)
        # predicted prior from dynamics
        prior = self._prior(feat['deter'])
        # actual posterior from observation
        post = feat['logit']
        # stop-grad posterior
        dyn = self._dist(sg(post)).kl(self._dist(prior))
        # stop-grad prior
        rep = self._dist(post).kl(self._dist(sg(prior)))
        if self.free_nats:
            dyn = jnp.maximum(dyn, self.free_nats)
            rep = jnp.maximum(rep, self.free_nats)
        losses = {'dyn': dyn, 'rep': rep}
        metrics['dyn_ent'] = self._dist(prior).entropy().mean()
        metrics['rep_ent'] = self._dist(post).entropy().mean()
        return carry, entries, losses, feat, metrics

    def trf_core(self, deter, stoch, action):
        # use Transformer for recurrent dynamics instead of GRU
        stoch = stoch.reshape((stoch.shape[0], -1))
        action /= sg(jnp.maximum(1, jnp.abs(action)))
        # prepare inputs for Transformer
        # project inputs to hidden dimension
        x0 = self.sub('dynin0', nn.Linear, self.hidden, **self.kw)(deter)
        x0 = nn.act(self.act)(self.sub('dynin0norm', nn.Norm, self.norm)(x0))
        x1 = self.sub('dynin1', nn.Linear, self.hidden, **self.kw)(stoch)
        x1 = nn.act(self.act)(self.sub('dynin1norm', nn.Norm, self.norm)(x1))
        x2 = self.sub('dynin2', nn.Linear, self.hidden, **self.kw)(action)
        x2 = nn.act(self.act)(self.sub('dynin2norm', nn.Norm, self.norm)(x2))

        # stack the inputs for the Transformer (concatenation messes up the shape)
        x = jnp.stack([x0, x1, x2], axis=1)
        g = self.attention_heads
        # x = jnp.concatenate([x0, x1, x2], -1)[..., None, :].repeat(g, -2)
        # x = self.sub('projection', nn.Linear, self.hidden, **self.kw)(x)
        # create mask for causal masking
        seq_len = x.shape[1]
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        mask = mask[None, :, :]
        mask = jnp.repeat(mask, x.shape[0], axis=0)
        # use transformer for recurrent processing
        x = self.sub('transformer', nn.Transformer,
                     units=self.hidden,
                     layers=self.trf_layers,
                     heads=self.attention_heads,
                     act=self.act,
                     norm=self.norm,
                     glu=True,
                     outscale=self.outscale)(x=x, ts=None,
                                             mask=mask, training=True)
        x = self.sub('proj_out', nn.Linear, self.deter)(x)
        # approach 1: mean pooling
        if not self.gating:
            deter = x.mean(axis=1) + deter
        # approach 2: trf gating
        else:
            x = x.reshape(x.shape[0], -1)
            flat2group = lambda x: einops.rearrange(x, '... (g h) -> ... g h', g=g)
            group2flat = lambda x: einops.rearrange(x, '... g h -> ... (g h)', g=g)
            gates = jnp.split(flat2group(x), 3, -1)
            reset, cand, update = [group2flat(x) for x in gates]
            reset = jax.nn.sigmoid(reset)
            cand = jnp.tanh(reset * cand)
            update = jax.nn.sigmoid(update - 1)
            deter = update * cand + (1 - update) * deter

        return deter

    def _prior(self, feat):
        # computes prior distribution logits
        x = feat
        for i in range(self.imglayers):
            x = self.sub(f'prior{i}', nn.Linear, self.hidden, **self.kw)(x)
            x = nn.act(self.act)(self.sub(f'prior{i}norm', nn.Norm, self.norm)(x))
        return self._logit('priorlogit', x)

    def _logit(self, name, x):
        # converts feature vector into logits over stochastic variables
        kw = dict(**self.kw, outscale=self.outscale)
        x = self.sub(name, nn.Linear, self.stoch * self.classes, **kw)(x)
        return x.reshape(x.shape[:-1] + (self.stoch, self.classes))

    def _dist(self, logits):
        # returns a categorical distribution over logits with uniform mixing
        out = embodied.jax.outs.OneHot(logits, self.unimix, self.adaptive_unimix)
        out = embodied.jax.outs.Agg(out, 1, jnp.sum)
        return out


class Encoder(nj.Module):
    units: int = 1024
    norm: str = 'rms'
    act: str = 'gelu'
    depth: int = 64
    mults: tuple = (2, 3, 4, 4)
    layers: int = 3
    kernel: int = 5
    symlog: bool = True
    outer: bool = False
    strided: bool = False
    attention_heads: int = 8

    def __init__(self, obs_space, **kw):
        # check that all input observations are at most 3d
        assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
        self.obs_space = obs_space
        # separate observation keys into vector and image inputs
        self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
        self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
        # define number of cnn layers
        self.depths = tuple(self.depth * mult for mult in self.mults)
        self.kw = kw

    @property
    def entry_space(self):
        # encoder produces no intermediate entries
        return {}

    def initial(self, batch_size):
        # encoder has no carry state
        return {}

    def truncate(self, entries, carry=None):
        # encoder has no carry state to truncate
        return {}

    def __call__(self, carry, obs, reset, training, single=False):
        # select shape index based on single/batched input
        bdims = 1 if single else 2
        outs = []
        bshape = reset.shape

        if self.veckeys:
            # process vector observations through mlp
            vspace = {k: self.obs_space[k] for k in self.veckeys}
            vecs = {k: obs[k] for k in self.veckeys}
            squish = nn.symlog if self.symlog else lambda x: x
            x = nn.DictConcat(vspace, 1, squish=squish)(vecs)
            x = x.reshape((-1, *x.shape[bdims:]))
            for i in range(self.layers):
                x = self.sub(f'mlp{i}', nn.Linear, self.units, **self.kw)(x)
                x = nn.act(self.act)(self.sub(f'mlp{i}norm', nn.Norm, self.norm)(x))

            # adding multi-head attention after MLP layers
            attention_module = self.sub('attn_vec', nn.Attention,
                                        heads=self.attention_heads,
                                        kv_heads=0, dropout=0.1)
            x = attention_module(x, training=training)
            outs.append(x)

        if self.imgkeys:
            # process image observations through convnet
            K = self.kernel
            imgs = [obs[k] for k in sorted(self.imgkeys)]
            assert all(x.dtype == jnp.uint8 for x in imgs)
            x = nn.cast(jnp.concatenate(imgs, -1), force=True) / 255 - 0.5
            x = x.reshape((-1, *x.shape[bdims:]))
            # no. of CNN layers
            for i, depth in enumerate(self.depths):
                if self.outer and i == 0:
                    # use non-strided conv on first layer if outer is set
                    x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
                elif self.strided:
                    # use strided conv if enabled
                    x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, 2, **self.kw)(x)
                else:
                    # otherwise use conv followed by 2x2 maxpool
                    x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
                    B, H, W, C = x.shape
                    x = x.reshape((B, H // 2, 2, W // 2, 2, C)).max((2, 4))

                # convnext block
                x = self.sub(f'convnext{i}', nn.ConvNeXtBlock, dim=depth)(x)
                x = nn.act(self.act)(self.sub(f'cnn{i}norm', nn.Norm, self.norm)(x))

            assert 3 <= x.shape[-3] <= 16, x.shape
            assert 3 <= x.shape[-2] <= 16, x.shape

            # adding multi-head attention after CNN layers
            # flatten spatial dimensions (H, W) into T
            x = x.reshape((x.shape[0], -1, x.shape[-1]))
            attention_module = self.sub('attn_img', nn.Attention,
                                        heads=self.attention_heads,
                                        kv_heads=0, dropout=0.1)
            x = attention_module(x, training=training)
            outs.append(x)

        # concatenate vector and image outputs
        x = jnp.concatenate(outs, -1)
        # reshape to match input batch dimensions
        tokens = x.reshape((*bshape, *x.shape[1:]))
        entries = {}
        return carry, entries, tokens


class Decoder(nj.Module):
    units: int = 1024
    norm: str = 'rms'
    act: str = 'gelu'
    outscale: float = 1.0
    depth: int = 64
    mults: tuple = (2, 3, 4, 4)
    layers: int = 3
    kernel: int = 5
    symlog: bool = True
    bspace: int = 8
    outer: bool = False
    strided: bool = False
    attention_heads: int = 8

    def __init__(self, obs_space, **kw):
        # check that all input observations are at most 3d
        assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
        self.obs_space = obs_space
        # separate observation keys into vector and image inputs
        self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
        self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
        # define number of channels per cnn block
        self.depths = tuple(self.depth * mult for mult in self.mults)
        # total number of image output channels
        self.imgdep = sum(obs_space[k].shape[-1] for k in self.imgkeys)
        # get image resolution from the first image key
        self.imgres = self.imgkeys and obs_space[self.imgkeys[0]].shape[:-1]
        self.kw = kw

    @property
    def entry_space(self):
        # decoder produces no intermediate entries
        return {}

    def initial(self, batch_size):
        # decoder has no carry state
        return {}

    def truncate(self, entries, carry=None):
        # decoder has no carry state to truncate
        return {}

    def __call__(self, carry, feat, reset, training, single=False):
        # check that deterministic feature dimension is divisible by bspace
        assert feat['deter'].shape[-1] % self.bspace == 0
        K = self.kernel
        recons = {}
        bshape = reset.shape

        # prepare feature input by flattening and concatenating
        inp = [nn.cast(feat[k]) for k in ('stoch', 'deter')]
        inp = [x.reshape((math.prod(bshape), -1)) for x in inp]
        inp = jnp.concatenate(inp, -1)

        if self.veckeys:
            # process vector targets through mlp and map to prediction heads
            spaces = {k: self.obs_space[k] for k in self.veckeys}
            o1, o2 = 'categorical', ('symlog_mse' if self.symlog else 'mse')
            outputs = {k: o1 if v.discrete else o2 for k, v in spaces.items()}
            kw = dict(**self.kw, act=self.act, norm=self.norm)
            x = self.sub('mlp', nn.MLP, self.layers, self.units, **kw)(inp)
            x = x.reshape((*bshape, *x.shape[1:]))
            kw = dict(**self.kw, outscale=self.outscale)
            outs = self.sub('vec', embodied.jax.DictHead, spaces, outputs, **kw)(x)
            recons.update(outs)

        if self.imgkeys:
            # compute shape of the spatial feature map before upsampling
            factor = 2 ** (len(self.depths) - int(bool(self.outer)))
            minres = [int(x // factor) for x in self.imgres]
            assert 3 <= minres[0] <= 16, minres
            assert 3 <= minres[1] <= 16, minres
            shape = (*minres, self.depths[-1])

            if self.bspace:
                # use structured projection to spatial feature map
                u, g = math.prod(shape), self.bspace
                x0, x1 = nn.cast((feat['deter'], feat['stoch']))
                x1 = x1.reshape((*x1.shape[:-2], -1))
                x0 = x0.reshape((-1, x0.shape[-1]))
                x1 = x1.reshape((-1, x1.shape[-1]))
                x0 = self.sub('sp0', nn.BlockLinear, u, g, **self.kw)(x0)
                x0 = einops.rearrange(x0, '... (g h w c) -> ... h w (g c)', h=minres[0], w=minres[1], g=g)
                x1 = self.sub('sp1', nn.Linear, 2 * self.units, **self.kw)(x1)
                x1 = nn.act(self.act)(self.sub('sp1norm', nn.Norm, self.norm)(x1))
                x1 = self.sub('sp2', nn.Linear, shape, **self.kw)(x1)
                x = nn.act(self.act)(self.sub('spnorm', nn.Norm, self.norm)(x0 + x1))
            else:
                # fallback to direct linear projection
                x = self.sub('space', nn.Linear, shape, **self.kw)(inp)
                x = nn.act(self.act)(self.sub('spacenorm', nn.Norm, self.norm)(x))

            # apply attention over spatial features [B, H, W, C] -> [B, H*W, C]
            B, H, W, C = x.shape
            x_flat = x.reshape((B, H * W, C))
            # apply attention
            attn = self.sub('attn_img', nn.Attention,
                            heads=self.attention_heads,
                            dropout=0.1)(x_flat, training=training)
            # residual connection and reshape back
            x = x_flat + attn
            x = x.reshape((B, H, W, C))

            # upsample spatial features using convtranspose or repetition
            for i, depth in reversed(list(enumerate(self.depths[:-1]))):
                if self.strided:
                    kw = dict(**self.kw, transp=True)
                    x = self.sub(f'conv{i}', nn.Conv2D, depth, K, 2, **kw)(x)
                else:
                    x = x.repeat(2, -2).repeat(2, -3)
                    x = self.sub(f'conv{i}', nn.Conv2D, depth, K, **self.kw)(x)

                # convnext block
                x = self.sub(f'convnext{i}', nn.ConvNeXtBlock, dim=depth)(x)
                x = nn.act(self.act)(self.sub(f'conv{i}norm', nn.Norm, self.norm)(x))

            # apply final output conv to generate pixel predictions
            if self.outer:
                kw = dict(**self.kw, outscale=self.outscale)
                x = self.sub('imgout', nn.Conv2D, self.imgdep, K, **kw)(x)
            elif self.strided:
                kw = dict(**self.kw, outscale=self.outscale, transp=True)
                x = self.sub('imgout', nn.Conv2D, self.imgdep, K, 2, **kw)(x)
            else:
                x = x.repeat(2, -2).repeat(2, -3)
                kw = dict(**self.kw, outscale=self.outscale)
                x = self.sub('imgout', nn.Conv2D, self.imgdep, K, **kw)(x)
            x = jax.nn.sigmoid(x)
            x = x.reshape((*bshape, *x.shape[1:]))

            # split final prediction into separate image keys
            split = np.cumsum(
                [self.obs_space[k].shape[-1] for k in self.imgkeys][:-1])
            for k, out in zip(self.imgkeys, jnp.split(x, split, -1)):
                out = embodied.jax.outs.MSE(out)
                out = embodied.jax.outs.Agg(out, 3, jnp.sum)
                recons[k] = out

        entries = {}
        return carry, entries, recons
