"""
MATCC / MASTER dataset classes.

Vendored verbatim from the MASTER fork of Qlib
(https://github.com/SJTU-Quant/MASTER -> qlib/contrib/data/dataset.py), because the
classes ``marketDataHandler`` and ``MASTERTSDatasetH`` are NOT present in the stock
qlib installed in the ``matcc`` conda env (qlib 0.9.7). Vendoring them here keeps the
environment clean (no qlib re-install) while letting the yaml reference them via
``module_path: "MATCC_dataset.py"``.

These two classes build the 63-dim market-guidance feature block (21 expressions x 3
market indices) and concatenate it with the 158 Alpha158 stock features, producing the
``[N, T, 222]`` tensors (158 stock + 63 market + 1 label) that the MATCC model expects.
"""

import copy

import pandas as pd
from typing import Union, List, Dict

from qlib.data.dataset import DatasetH, TSDatasetH, TSDataSampler
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import check_transform_proc


###################################################################################
# lqa: for MASTER
class marketDataHandler(DataHandlerLP):
    """Market Data Handler for MASTER (see `examples/benchmarks/MASTER`)

    Args:
        instruments (str): instrument list
        start_time (str): start time
        end_time (str): end time
        freq (str): data frequency
        infer_processors (list): inference processors
        learn_processors (list): learning processors
        fit_start_time (str): fit start time
        fit_end_time (str): fit end time
        process_type (str): process type
        filter_pipe (list): filter pipe
        inst_processors (list): instrument processors
    """
    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=[],
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        market_indices=None,
        **kwargs
    ):
        # The 3 market indices feeding MASTER's market-guided gate. Default = CN CSI300/100/500.
        # For us_data pass e.g. ["^gspc", "^dji", "^ndx"] to keep the same 63-dim structure.
        self.market_indices = market_indices or ["sh000300", "sh000903", "sh000905"]
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs
        )

    def get_feature_config(self):
        """Market features (63-dim = 21 formulas x 3 indices).

        Default indices are the CN CSI300/100/500 (MASTER's original setup). Override via
        ``market_indices`` (e.g. ["^gspc", "^dji", "^ndx"] for US) to keep the same
        63-dim structure without changing the model architecture.

        Returns:
            (fields, names): identical lists of Mask(...) expressions.
        """
        # 21 market-feature expression bodies; the instrument is applied via Mask below.
        exprs = ["$close/Ref($close,1)-1"]
        for w in (5, 10, 20, 30, 60):
            exprs += [
                f"Mean($close/Ref($close,1)-1,{w})",
                f"Std($close/Ref($close,1)-1,{w})",
                f"Mean($volume,{w})/$volume",
                f"Std($volume,{w})/$volume",
            ]
        fields = []
        for inst in self.market_indices:
            fields += [f'Mask({e}, "{inst}")' for e in exprs]
        return fields, list(fields)


class MASTERTSDatasetH(TSDatasetH):
    """
    MASTER Time Series Dataset with Handler

    Args:
        market_data_handler_config (dict): market data handler config
    """
    def __init__(
        self,
        market_data_handler_config = Dict,
        **kwargs,
    ):
        super().__init__(**kwargs)
        marketdl = marketDataHandler(**market_data_handler_config)
        self.market_dataset = DatasetH(marketdl, segments = self.segments)


    def get_market_information(
        self,
        slc: slice,
    ) -> Union[List[pd.DataFrame], pd.DataFrame]:
        return self.market_dataset.prepare(slc)

    def _prepare_seg(self, slc: slice, **kwargs) -> TSDataSampler:
        dtype = kwargs.pop("dtype", None)
        if not isinstance(slc, slice):
            slc = slice(*slc)
        start, end = slc.start, slc.stop
        flt_col = kwargs.pop("flt_col", None)
        # TSDatasetH will retrieve more data for complete time-series

        ext_slice = self._extend_slice(slc, self.cal, self.step_len)
        only_label = kwargs.pop("only_label", False)
        data = super(TSDatasetH, self)._prepare_seg(ext_slice, **kwargs)

        ############################## Add market information ###########################
        # If we only need label for testing, we do not need to add market information
        if not only_label:
            marketData = self.get_market_information(ext_slice)
            cols = pd.MultiIndex.from_tuples([("feature", feature) for feature in marketData.columns])
            marketData = pd.DataFrame(marketData.values, columns = cols, index = marketData.index)
            data = data.iloc[:,:-1].join(marketData).join(data.iloc[:,-1])
        #################################################################################
        flt_kwargs = copy.deepcopy(kwargs)
        if flt_col is not None:
            flt_kwargs["col_set"] = flt_col
            flt_data = super()._prepare_seg(ext_slice, **flt_kwargs)
            assert len(flt_data.columns) == 1
        else:
            flt_data = None

        tsds = TSDataSampler(
            data=data,
            start=start,
            end=end,
            step_len=self.step_len,
            dtype=dtype,
            flt_data=flt_data,
            fillna_type = "ffill+bfill"
        )
        return tsds
