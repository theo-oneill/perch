import queue
import time
from cube import Cube
from write_pairs import WritePairs
from coboundary_enumerator import CoboundaryEnumerator

class ComputePairs:
    def __init__(self, dcg, wp, config):
        self.dcg = dcg
        self.dim = 1  # default method is LINK_FIND, where we skip dim=0
        self.wp = wp
        self.config = config
        self.pivot_column_index = {}

    def compute_pairs_main(self, ctr):
        coface_entries = []
        ctl_size = len(ctr)
        num_apparent_pairs = 0

        if self.config.verbose:
            print("# columns to reduce:", ctl_size)

        self.pivot_column_index.clear()
        recorded_wc = {}
        cached_column_idx = queue.Queue()
        recorded_wc = {}

        for i in range(ctl_size):
            working_coboundary = queue.Queue()
            birth = ctr[i].birth

            j = i
            pivot = None
            might_be_apparent_pair = True
            found_persistence_pair = False
            num_recurse = 0

            while True:
                cache_hit = False

                if i != j:
                    if j in recorded_wc:
                        cache_hit = True
                        wc = recorded_wc[j]
                        while not wc.empty():
                            working_coboundary.put(wc.get())

                if not cache_hit:
                    coface_entries.clear()
                    cofaces = CoboundaryEnumerator(self.dcg, self.dim)
                    cofaces.setCoboundaryEnumerator(ctr[j])
                    while cofaces.hasNextCoface():
                        coface_entries.append(cofaces.nextCoface)
                        if might_be_apparent_pair and ctr[j].birth == cofaces.nextCoface.birth:
                            if cofaces.nextCoface.index not in self.pivot_column_index:
                                pivot = cofaces.nextCoface
                                found_persistence_pair = True
                                break
                            else:
                                might_be_apparent_pair = False

                    if found_persistence_pair:
                        self.pivot_column_index[pivot.index] = i
                        num_apparent_pairs += 1
                        break

                    for e in coface_entries:
                        working_coboundary.put(e)

                pivot = self.get_pivot(working_coboundary)
                if pivot.index is not None:
                    if pivot.index in self.pivot_column_index:
                        j = self.pivot_column_index[pivot.index]
                        num_recurse += 1
                        continue
                    else:
                        if num_recurse >= self.config.min_recursion_to_cache:
                            self.add_cache(i, working_coboundary, recorded_wc)
                            cached_column_idx.put(i)
                            if cached_column_idx.qsize() > self.config.cache_size:
                                del recorded_wc[cached_column_idx.get()]

                        self.pivot_column_index[pivot.index] = i
                        death = pivot.birth
                        if birth != death:
                            self.wp.append(WritePairs(self.dim, birthC = ctr[i], deathC = pivot, dcg = self.dcg, print_flag = self.config.print))
                        break
                else:
                    if birth != self.dcg.threshold:
                        self.wp.append(WritePairs(self.dim, birth=birth, death=self.dcg.threshold, birth_x=ctr[i].x, birth_y=ctr[i].y, birth_z=ctr[i].z, death_x=0, death_y=0, death_z=0, print_flag=self.config.print))
                    break

        if self.config.verbose:
            print("# apparent pairs:", num_apparent_pairs)

    def add_cache(self, i, wc, recorded_wc):
        clean_wc = queue.Queue()
        while not wc.empty():
            c = wc.get()
            if not wc.empty() and c.index == wc.queue[0].index:
                wc.get()
            else:
                clean_wc.put(c)
        recorded_wc[i] = clean_wc

    def pop_pivot(self, column):
        if column.empty():
            return None
        else:
            pivot = column.get()

            while not column.empty() and column.queue[0].index == pivot.index:
                column.get()
                if column.empty():
                    return None
                else:
                    pivot = column.get()
            return pivot

    def get_pivot(self, column):
        pivot = self.pop_pivot(column)
        if pivot is not None:
            column.put(pivot)
        return pivot

    def assemble_columns_to_reduce(self, ctr, dim):
        self.dim = dim
        ctr.clear()
        max_m = 3
        if dim == 0:
            max_m = 1
            self.pivot_column_index.clear()

        for m in range(max_m):
            for z in range(self.dcg.az):
                for y in range(self.dcg.ay):
                    for x in range(self.dcg.ax):
                        birth = self.dcg.get_birth(x, y, z, m, dim)
                        v = Cube(birth, x, y, z, m)
                        if birth < self.dcg.threshold and v.index not in self.pivot_column_index:
                            ctr.append(v)

        start = time.time()
        ctr.sort(key=lambda c: c.birth)
        if self.config.verbose:
            end = time.time()
            print("Sorting took:", (end - start) * 1000, "milliseconds")