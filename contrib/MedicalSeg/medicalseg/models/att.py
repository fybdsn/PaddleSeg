from operator import itemgetter
from paddle import nn
import paddle


def map_el_ind(arr, ind):
    return list(map(itemgetter(ind), arr))


def sort_and_return_indices(arr):
    indices = [ind for ind in range(len(arr))]
    arr = zip(arr, indices)
    arr = sorted(arr)
    return map_el_ind(arr, 0), map_el_ind(arr, 1)


class SelfAttention(nn.Layer):
    def __init__(self, dim, heads, dim_heads = None):
        super().__init__()
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads

        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden)
        self.to_kv = nn.Linear(dim, 2 * dim_hidden)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x, kv = None):
        kv = x if kv is None else kv

        q, k, v = (self.to_q(x), *paddle.chunk(self.to_kv(kv),2,axis = -1))

        b, t, d, h, e = *q.shape, self.heads, self.dim_heads

        merge_heads = lambda x: paddle.reshape(paddle.reshape(x,(b, -1, h, e)).transpose((0,2,1,3)),(b * h, -1, e))
        q, k, v = map(merge_heads, (q, k, v))

        dots = paddle.einsum('bie,bje->bij', q, k) * (e ** -0.5)
        # dots = dots.Softmax(dim=-1)
        dots = nn.Softmax(axis = -1)(dots)
        out = paddle.einsum('bij,bje->bie', dots, v)

        # out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = paddle.reshape(out,(b, h, -1, e)).transpose((0,2,1,3))
        out = paddle.reshape(out,(b, -1, d))
        out = self.to_out(out)
        return out

class PermuteToFrom(nn.Layer):
    def __init__(self, permutation, fn):
        super().__init__()
        self.fn = fn
        _, inv_permutation = sort_and_return_indices(permutation)
        self.permutation = permutation
        self.inv_permutation = inv_permutation

    def forward(self, x, **kwargs):

        # axial = x.permute(*self.permutation).contiguous()
        axial = paddle.transpose(x,self.permutation)
        shape = axial.shape
        *_, t, d = shape

        # merge all but axial dimension
        # axial = axial.reshape(-1, t, d)
        axial = paddle.reshape(axial,(-1,t,d))
        # attention
        axial = self.fn(axial, **kwargs)

        # restore to original shape and permutation
        axial = axial.reshape(shape)
        axial = axial.transpose(self.inv_permutation)
        return axial

def calculate_permutations(num_dimensions, emb_dim):
    total_dimensions = num_dimensions + 2
    emb_dim = emb_dim if emb_dim > 0 else (emb_dim + total_dimensions)
    axial_dims = [ind for ind in range(1, total_dimensions) if ind != emb_dim]

    permutations = []

    for axial_dim in axial_dims:
        last_two_dims = [axial_dim, emb_dim]
        dims_rest = set(range(0, total_dimensions)) - set(last_two_dims)
        permutation = [*dims_rest, *last_two_dims]
        permutations.append(permutation)

    return permutations


class AxialAttention(nn.Layer):
    def __init__(self, dim, num_dimensions = 2, heads = 8, dim_heads = None, dim_index = -1, sum_axial_out = True):
        assert (dim % heads) == 0, 'hidden dimension must be divisible by number of heads'
        super().__init__()
        self.dim = dim
        self.total_dimensions = num_dimensions + 2
        self.dim_index = dim_index if dim_index > 0 else (dim_index + self.total_dimensions)

        attentions = []
        for permutation in calculate_permutations(num_dimensions, dim_index):
            attentions.append(PermuteToFrom(permutation, SelfAttention(dim, heads, dim_heads)))

        self.axial_attentions = nn.LayerList(attentions)
        self.sum_axial_out = sum_axial_out

    def forward(self, x):
        assert len(x.shape) == self.total_dimensions, 'input tensor does not have the correct number of dimensions'
        assert x.shape[self.dim_index] == self.dim, 'input tensor does not have the correct input dimension'

        if self.sum_axial_out:
            return sum(map(lambda axial_attn: axial_attn(x), self.axial_attentions))

        out = x
        for axial_attn in self.axial_attentions:
            out = axial_attn(out)
        return out

if __name__ == '__main__':

    x = paddle.randn((1, 16, 16, 32, 32))
    attn = AxialAttention(
        dim = 16,  # embedding dimension
        dim_index = 1,  # where is the embedding dimension
        heads = 2,  # number of heads for multi-head attention
        num_dimensions = 3,  # number of axial dimensions (images is 2, video is 3, or more)
    )
    o = attn(x)
    print(o.shape)
