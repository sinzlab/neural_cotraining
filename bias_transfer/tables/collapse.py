import nnfabrik.utility.dj_helpers as util
import datajoint as dj


class Collapsed(dj.Computed):
    Source = None

    @property
    def definition(self):
        return """
        collapsed_key:      varchar(64)
        ---
        -> self.Source{}
       """.format(".proj(collapsed_key_='collapsed_key')" if hasattr(self.Source, "transfer_steps") else "")

    @property
    def key_source(self):
        if hasattr(self.Source, "transfer_steps"):
            return self.Source.proj(collapsed_key_='collapsed_key')
        else:
            return self.Source

    @property
    def target(self):
        if hasattr(self.Source, "transfer_steps"):
            return self.Source.proj(collapsed_key_='collapsed_key') & self
        else:
            return self.Source & self

    def make(self, key):
        hash = util.make_hash(key)
        key['collapsed_key'] = hash
        self.insert1(key)
