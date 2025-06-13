from collections import deque
import xxhash
import numpy as np

from tinyvllm.engine.sequence import Sequence

# We take prefix hash with new sequences to avoid hash collisions
def compute_hash(token_ids: list[int], prefix: int = -1):
    h = xxhash.xxh64()
    if prefix != -1:
       h.update(prefix.to_bytes(8, 'little')) 
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()


class Block:
    
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = - 1
        self.token_ids = []
        
    def update(self, hash: int, token_ids: list[int]):
        assert hash != -1, "Hash must not be -1"
        self.hash = hash
        self.token_ids = token_ids
        
    def reset(self):
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []
        
    def __repr__(self):
        return f"{(self.block_id, self.ref_count, self,hash)}"
    

class BlockManager:
    
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        
    def _allocate_block(self, block_id: int):
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]
    
    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0, "Block must not be in use"
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)
        
    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1 
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            
        