#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions for examining seasonality and seasonal correlations in OHLC
financial time series.

Includes function for correcting DST shift before analysis.

TODO: 
Remove the import bloat.
Ready module to be incorporated into a program. This entails:
Abandon DF metadata. That stuff can be coded much easier in a wrapper
program.
Break up CandleTrans. Make candlestats work with any input.
Write unit tests.
Move useful stuff from backtest module, most noteably tick2ohlc.
Consider using a plotting function from R or Java. matplotlib = meh.
Consider inverting your colorschemes for web display.

LINE PROFILES ARE NO LONGER ACCURATE!
"""

# import modules   ##############################
from __future__ import division  # So that 1/2 = 0.5
try:
    from scipy.interpolate import interp1d
    import pandas as pd
    import numpy as np
    import datetime as dt
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy.ma as ma
    import matplotlib.cm as cm
    import time
    import string
    import math
    import pytz
    import sys
except:
    print "[ERROR] One or more of the required modules cannot be loaded."
    sys.exit(0)


# IMPORTANT:
# Convention: The name of a candle is the time at which it closes.
# The 6:00 candle closes at 6:00 whether you are looking at hour candles or
# minute candles.  The hour candle contains 4 * 15 min candles:
# the 5:15, 5:30, 5:45, 6:00.
# These candles are shown graphically as fitting between the 5:00 and 6:00
# ticks.
# For resampling, time bins are labeled with the right side value.
# (NOT LEFT SIDE!!!!!!)
# Time bins must be closed on the right. (left = upsample bfill will fail.)
#
# NOTE: THIS IS FUCKING WRONG
#
#
#
# Load price data from file ######################


def load_ohlc(filename):
    """
    #Load raw pice data from file into dataframe called raw_price.
    #At present, Does not tolerate column header line.
    #
    # TODO:
    # 0: I open the file at the beginning but never close it at the end.
    # 1: One year of candles takes 33 seconds. Implement some kind of
    #    progress bar based on file size. Needs major rewrite to load in
    #    chuncks instead of all at once. This woud allow progress bar.
    #    unfortunately this means concatenating
    #    a df over and over, which I think means copying each time.
    # 2: Parse based on header line if exists or ignore header line if not.
    # 3: Include some kind of error handling. inc: empty file, High and Low not
    #    actually high and low, gappy data, multiples, etc.
    # 4: Even better, autodetect OHLC. Should be simple to figure out from
    #    just 10 or so lines. If can't autodetect, throw excpetion.
    # The data I downloaded first has many 1 pip gaps between close and open.
    # Do a stats printout with % gaps, average open-close gap size, % missing
    # data.
    # Add option to load tick data as well. Resample to 1 second?
    # or, instead make separate load_tick() function. That might make more
    # sense. Then I could
    # include tick-ohlc conversion there. might make more sense.
    # 5: Convert column names to lower case as to be compatible with standard
    # pandas OHLC scheme. (You sure about this?)
    # All code needs to be refactored this way.
    """
    start_time = time.time()
    if '/' in filename:
        dataset_name = string.split(filename, '/')[-1][:-4]
    else:
        dataset_name = string.split(filename, '\\')[-1][:-4]
    # Lame system specific hack.
    fp = open(filename)
    num_lines = sum(1 for line in fp)
    fp.close()
    print '\rEstimated loading time: %d seconds' % \
        (num_lines * 9.038440596754874e-05)
    # load and parse
    raw_price = pd.read_csv(filename,
                            names=['Inst.', 'Date', 'Time', 'Open',
                                   'High', 'Low', 'Close', 'Volume'],
                            index_col='Date_Time',
                            parse_dates=[[1, 2]],
                            header=None,
                            skipinitialspace=True)
    del raw_price['Volume']
    del raw_price['Inst.']
    # Rename columns if needed.
    # raw_price.columns = ['Inst.', 'Open', 'High', 'Low', 'Close', 'Vol']
    raw_price = raw_price.tz_localize('UTC')
    # Change later: raw_price = raw_price.tz_convert('UTC')
    time_diff = str(round(time.time() - start_time, 1))
    print '\r%d rows from %s loaded in %s seconds' % (len(raw_price),
                                                      dataset_name,
                                                      time_diff)
    return raw_price


def down_sample(prices, offset):
    """
    Resamples raw OHLC price data to the frequency specified in 'offset'.

    Frequency is given in minutes by default. If offset is non-numeric,
    uses standard pandas offset aliases.
    http://pandas.pydata.org/pandas-docs/dev/timeseries.html#offset-aliases
    # TODO:
    # Put in something so that extra columns get passed somehow. Not sure how
    # to do this. If not, put in a warning that says there are extra columns
    # that
    # will be deleted. Some resamaples should be forbidden. eg: resample 10min
    # to 15 min
    # since they aren't multiples and information will be lost.
    # Find a way to deal with incomplete or unfinished candles.
    # Answer: add boolean column called 'complete'
    # (This wa implemented in the backtest version of down_sample.)
    # Change to lower case, or make option to use both.
    """
    ohlc_dict = {'Open': 'first',
                 'High': 'max',
                 'Low': 'min',
                 'Close': 'last'}
    offset = str(offset)
    if offset.isdigit():
        offset = str(offset) + 'Min'
    # prices = price.drop(price.index[0])
    # To drop multiples: df.drop(df.index[[1, 3]])
    result = prices.resample(offset,
                             how=ohlc_dict,
                             closed='right',
                             label='right')
    return result


def dst_slice(ohlc_series, EDT=None, BST=None, label=True, verbose=False, clean=False):
    """
    Takes OHLC time series and labels by DST status.

    If EDT (eastern daylight Time) and/or BST (British Summer Time) are given
    as arguments (True or False), the output data are selected by these
    conditons. If verbose=True, then DST periods are given and the user is
    prompted to select one for output. (This last option is not yet fully
    implemented.)
    TO DO:
    # 2: Could be much faster: does 1 year of 5Min data in 8.9 seconds.
    #    This got much slower in pandas 12/13. Not sure why: 405 seconds.
    #    Doing this on a 1 hr resampled copy will save on lots of time. Then
    #    resample back to original and combine to the two DFs? Or, do it like
    #    you did classify_ohlc.
    # 4: Finish verbose option to select all data, or just certain (1 or more)
    #    DST sections.
    # 5: Give option to save?
    # 6: This module should be the one to start propagating metadata.
    # 6: Give option to remove change-over weeks and convert True/True regions
    #    to BST or whatever. This would closely approximate what is seen in
    #    the US at the expense of losing four weeks and losing accuracy from
    #    coutries not observing nothern hemisphere DST.
    # 7: Need to test all of this with unit tests.
    """
    start_time = time.time()
    new_york_zone = pytz.timezone('US/Eastern')
    london_zone = pytz.timezone('Europe/London')
    # Create a copy and make a column with index cast as datetime.datetime
    prices = ohlc_series.copy()
    prices['Times'] = prices.index.to_pydatetime()
    if label:  # Label the candles if requested in fn call
        prices['USDST'] = True  # initialize column to boolean
        prices['EUDST'] = True  # initialize column to boolean
    # Loop through and find DST for each entry using pytz
    print "Labeling candles by DST status..."
    dst_period_counter = 1
    dst_period = []
    dst_period_list = []
    transition_time = prices.index[0]  # Set initial date
    old_london_test = (transition_time.astimezone(
        london_zone).dst().seconds // 3600 == 1)  # Error? .total_seconds()
    old_newyork_test = (transition_time.astimezone(
        new_york_zone).dst().seconds // 3600 == 1)  # Error? .total_seconds()
    dst_period = [prices.Times.ix[0], old_london_test, old_newyork_test]
    counter_max = int(len(prices))  # This is my completion counter max
    for i in xrange(0, counter_max):  # xrange not faster, maybe for large sets?
        transition_time = prices.Times.ix[i]  # 287us
        london_test = (transition_time.astimezone(
            london_zone).dst().seconds // 3600 == 1)  # Err? .total_seconds()
        newyork_test = (transition_time.astimezone(
            new_york_zone).dst().seconds // 3600 == 1)  # Err? .total_seconds()
        if label:  # Label the candles if requested in fn call
            prices.USDST.ix[i] = newyork_test  # SLOW: 1.54 ms
            prices.EUDST.ix[i] = london_test  # SLOW: 1.54 ms
        switch = ((london_test == old_london_test) &
                  (newyork_test == old_newyork_test) != True)
        if switch:
            dst_period.append(prices.Times.ix[i])
            if verbose:
                print dst_period
            dst_period_list.append(dst_period[:])
            dst_period = [prices.Times.ix[i], london_test, newyork_test]
            dst_period_counter = dst_period_counter + 1
        old_london_test = london_test
        old_newyork_test = newyork_test
        progress = (100 * i // counter_max)
        sys.stdout.write("\rPercent complete: %d%%   " % (progress))
        sys.stdout.flush()
    dst_period.append(prices.Times.ix[
                      -1])  # put the last value in dst_period
    if verbose:
        print dst_period
    dst_period_list.append(dst_period[
                           :])  # Append the last date to the series list
    time_diff = str(round(time.time() - start_time, 1))
    print '\r%d candles checked in in  %s seconds' % (len(prices), time_diff)
    print '\rThere are %s DST periods in this sample' % (dst_period_counter)
    if ((verbose == True) & (dst_period_counter > 1)):
        var = raw_input("Select subseries to return: ")
        var = int(var) - 1
    if EDT:
        prices = prices[(prices.USDST == True)]
    elif EDT == False:
        prices = prices[(prices.USDST == False)]
    if BST:
        prices = prices[(prices.EUDST == True)]
    elif BST == False:
        prices = prices[(prices.EUDST == False)]
    del prices['Times']  # remove unneeded column
    if label == False:
        del prices['USDST']
        del prices['EUDST']
    return prices


def classify_ohlc(ohlc_series, boolean=True):
    """
    Classifies candles as belonging to an up or down hour candle.
    Classifies candles as belonging to an up or down day candle.
    Creates two new boolean columns. At this point, input can only
    be an OHLC series with no other columns. This is because the series
    needs to be resampled. It's frequency must be 60Min or less.
    Most things I can think of that would use ths function would benefit
    from not having weekends included. However, to make it as useful as
    possible,
    this function returns a series of the same shape as the input.
    If boolean argument is False then recast boolean results columns as
    float.
    # TO DO:
    # Do I need to resample the input?
    # How to deal with partial days and partial hours?
    # Somehow insist that the original DF is the one with the "right" index?
    # 1: make sure to put the freq back to the original.
    # on second, second thought, put in somethng that prevents FiveSeries
    if interval
    # is more than 5, etc...
    """
    start_time = time.time()
    prices = ohlc_series.copy()
    zone = prices.index.tz.zone
    # determine frequency. Later use a cond?: if prices.index.freq != None
    tsInterval = prices.index[1] - prices.index[0]
    interval = tsInterval.seconds // 60  # Error? .total_seconds()
    intervalStr = str(int(interval)) + 'Min'
    # prices = down_sample(prices, (tsInterval.seconds // 60))
    # remove gaps, Error? I think this might be .total_seconds()

    FiveSeries = down_sample(prices, 5).dropna()
    QSeries = down_sample(prices, 15).dropna()
    HourSeries = down_sample(prices, 60).dropna()
    DaySeries = down_sample(prices, 1440).dropna()

    FiveSeries['FiveGain'] = (FiveSeries['Close'] >= FiveSeries['Open'])
    QSeries['QGain'] = (QSeries['Close'] >= QSeries['Open'])
    HourSeries['HourGain'] = (HourSeries['Close'] >= HourSeries['Open'])
    DaySeries['DayGain'] = (DaySeries['Close'] >= DaySeries['Open'])

    # Upsample back to original frequency, backfilling hour and day values.
    FiveToMin = FiveSeries.FiveGain.resample(intervalStr,
                                             fill_method='bfill',
                                             closed='right',
                                             label='right')
    QToMin = QSeries.QGain.resample(intervalStr,
                                    fill_method='bfill',
                                    closed='right',
                                    label='right')
    HourToMin = HourSeries.HourGain.resample(intervalStr,
                                             fill_method='bfill',
                                             closed='right',
                                             label='right')
    DayToMin = DaySeries.DayGain.resample(intervalStr,
                                          fill_method='bfill',
                                          closed='right',
                                          label='right')

    # Combine your new columns into the initial dataframe.
    pieces = [FiveToMin, QToMin, HourToMin, DayToMin]
    Both = pd.concat(pieces, axis=1)
    # problem above: creating a ts. loses column title, casts as float. Fix!
    Both.columns = ['FiveGain', 'QGain', 'HourGain', 'DayGain']
    if boolean:  # sketchy name.
        Both['FiveGain'] = Both['FiveGain'].astype(bool)
        Both['QGain'] = Both['QGain'].astype(bool)
        Both['HourGain'] = Both['HourGain'].astype(bool)
        Both['DayGain'] = Both['DayGain'].astype(bool)

    # Both references times that don't exist in the original ohlc series.
    # Now remove them:
    results = pd.concat([prices, Both], axis=1, join='inner')
    # Above join='inner' means index intersection
    results = results.dropna()
    # Output at this point is satisfactory, though not necessarily the same
    # form
    # as input with regard to weekends. The following was my attempt to fix
    # this
    # by concat-ing with the original df and then dropping duplicates.
    # This feature is not worth the effort right now.
    # results = pd.concat([prices, results], axis=0)
    # Now drop duplicates.
    # results["index"] = results.index
    # results.drop_duplicates(cols='index', take_last=True, inplace=True)
    # del results["index"]
    results = results.tz_localize('UTC')
    results = results.tz_convert(zone)
    return results


def CandleTrans(ohlc_series):
    """
    # Does candle by candle analysis of OHLC time series. Works quickly.
    #
    # TO DO:
    # 1: Fix NAN chaos.
    # 2: Break it up into smaller functions? Would that really be better?
    # 3: Make actual volatility stdev of return? or of the high and low?
    # annualize it? Express as a percent of price? Hmmm.
    """
    start_time = time.time()
    prices = ohlc_series.copy()
    tsInterval = prices.index[1] - prices.index[0]  # Warning!!!!
    interval = tsInterval.seconds // 60  # Error? .total_seconds()
    StrInterval = str(int(interval)) + 'Min'

    # First section: For functions requiring orginal index ###########

    # Make: DeltaPips column,
    prices['DeltaPips'] = 10000 * (prices['Close'] -
                                   prices['Open'])

    # Make:	AveMove=Abs(DeltaPips) column,
    prices['AbsPips'] = abs(prices['DeltaPips'])

    # Make: Total range column
    prices['PipsRange'] = abs(10000 * (prices['High'] -
                                       prices['Low']))
    # Make simple returns.
    prices['SimpRet'] = (prices['Close'] / prices['Open']) - 1

    # Make: log returns (Be sure float division is working.)
    prices['LogRet'] = log(prices['Close'] / prices['Open'])

    def FuzPips(A):  # Returns zeros NOT NaNs! Rows MUST be removed later!
        FuzPips = np.zeros(np.shape(A))  # Empty array in the shape of A
        for i, a in enumerate(A):
            if a < 0:
                FuzPips[i] = 0
            elif a > 0:
                FuzPips[i] = 1
            else:
                FuzPips[i] = 0.5
        return(FuzPips)

    prices['FzPips'] = FuzPips(prices['DeltaPips'])

    # PastTrend is like Trend, except it looks at past 2 candles instead of
    # previous, current and future. This way, it can be used in correlations
    # without future bias.
    # It can be thought of as trend in any direction vs no trend at all
    # (which could
    # simply mean a trend and then a stall.
    prices['PastTrend'] = ((prices['FzPips'] != 0.5) &
                         (((prices['FzPips'] == prices['FzPips'].shift(1)) &
                           (prices['FzPips'] == prices['FzPips'].shift(2)))))
    prices['PastTrend'] = prices['PastTrend'].astype(float)

    prices = prices.dropna(axis=0)
    # Resample Section #####
    prices = prices.resample(StrInterval)  # Puts the empty weekends back
    time_diff = str(round(time.time() - start_time, 1))
    print '\r%d lines done in in  %s seconds' % (len(prices), time_diff)
    return prices


def CandleStats(ohlc_series):
    """
    Does daily stats on DF returned by CandleTrans.
    Expects gapless dataframe. Discovers frequency by itself.
    Output dataframe has FloatTime column for later indexing and then
    plotting.
    Column: 'metarange' contains TS start and end in [0] and [1] for passing
    to next funtions.

    # TO DO:
    # 4: Break this function up into pieces. That's probably the only way that
    I will be able to pass metadata on to subsequent tools. When you break it
    up, you will
    # need to include another metadata column that says what the data is
    called and
    # what column it's in.
    # 5: Contains many inefficient df.set_value. Remove them?
    # # Note that there will be missing data on the right side of the graph.
    # This is because 00:00 = 24:00 because you are plotting where='pre', the
    00:00 data
    # falls off the lef side of the graph.
    # To fix this, you should move the first entry in the time series to the
    end
    and call it
    # 1970-01-01 23:59:59..... If it works, you should propagate this to any
    other
    # functions that need it.
    # But, if I remove the first entry, I need to make sure that any graph
    axis
    I make from
    # this is extended in the left direction as well.
    # Use this:
    # xlimits = ax1.get_xlim()
    # ax1.set_xlim(int(xlimits[0]) + 0.000000001, (int(xlimits[0])+1))
    # remove Timestamp column!
    """
    ResampPrice = ohlc_series.copy()  # now modifies copy, not original.
    start_time = time.time()
    tsInterval = ResampPrice.index[1] - ResampPrice.index[0]
    if tsInterval.seconds / 60 > 60:  # Error? .total_seconds()
        raise Exception("Time series frequency must be <= 60 minutes.")
    if 60 % (tsInterval.seconds / 60) != 0:  # Error?  .total_seconds()
        raise Exception("Frequency must be 60 or integer factor of 60.")
    interval = tsInterval.seconds // 60  # Error? .total_seconds()
    bins = 24 * (60 // interval)  # Keep integer division here.
    Results = pd.DataFrame(range(0, bins))
    # Create indices for later plotting
    intervalStr = str(int(interval)) + 'Min'
    Results['Time'] = pd.date_range('1970-01-01', '1970-01-01 23:59:59',
                                    freq=intervalStr)
    # Some functions below don't tolerate NANs. Get rid of them.
    ResampPrice = ResampPrice.dropna(axis=0)
    # Now do main analyis loop
    for num in range(0, bins):
        LoopResampPrice = ResampPrice[(((ResampPrice.index.hour *
                                         (60 / interval) +
                                         (ResampPrice.index.minute //
                                          interval))) == (num))]
        Results = Results.set_value(
            num, 'DPips', average(LoopResampPrice.DeltaPips))
        Results = Results.set_value(
            num, 'DPipsStd', std(LoopResampPrice.DeltaPips))
        Results = Results.set_value(
            num, 'AbsPips', average(LoopResampPrice.AbsPips))
        Results = Results.set_value(
            num, 'AbsPipsStd', std(LoopResampPrice.AbsPips))
        Results = Results.set_value(
            num, 'PipsRange', average(LoopResampPrice.PipsRange))
        Results = Results.set_value(
            num, 'PipsRangeStd', std(LoopResampPrice.PipsRange))
        Results = Results.set_value(
            num, 'FzPips', average(LoopResampPrice.FzPips))
        Results = Results.set_value(
            num, 'PastTrend', average(LoopResampPrice.PastTrend))
        Results = Results.set_value(
            num, 'SimpRet', average(LoopResampPrice.SimpRet))
        Results = Results.set_value(
            num, 'LogRet', average(LoopResampPrice.LogRet))
        progress = (100 * num // bins)
        sys.stdout.write("\rPercent complete: %d%%   " % (progress))
        sys.stdout.flush()

    # Now move the first entry to the end and call it: 1970-01-01 23:59:59
    Results = Results.append(Results.ix[0], ignore_index=True)
    Results = Results.drop(Results.index[0])
    # The index has just grown by 1. df.ix is now broken.
    # Great to change the TimeStamp but I don't think it's allowed
    # Do it after FloatTime is created.
    Results['FloatTime'] = mdates.date2num(Results.Time)
    Results.FloatTime[bins] = 719163.9999999
    # Remove TimeStamp index since it's of little use and broken anyway?
    # Write metadata: safe as long as you keep integer index
    # Results = Results.set_value(0, 'metaRange', ResampPrice.index[0])
    # Results = Results.set_value(1, 'metaRange', ResampPrice.index[-1])
    time_diff = str(round(time.time() - start_time, 1))
    print '\rCompleted in %s seconds' % time_diff
    return Results


def Zebra(target, ZebraNum):
    xlimits = target.get_xlim()
    xStart = xlimits[0]
    xEnd = xlimits[1]
    ZebraNum = ZebraNum / 2
    xLength = xEnd - xStart
    ZebraWidth = (xLength / ZebraNum) / 2
    i = xStart
    # i = i + (xLength/ZebraNum) # Uncomment to start color shading second bar
    while i < xEnd:
        target.axvspan(i, i + ZebraWidth, facecolor='0.2', alpha=0.05)
        i = i + (xLength / ZebraNum)
    return target


# Plot your candleStats ##############################
# def plot_stats(my_df):


# Format for plotting ####################################
# def pltformater(Results, interval):
#    """
#    TO DO: Put this into CandleStats
#    """
#    interval = str(int(interval)) + 'Min'
#    Results['Time'] = pd.date_range('1970-01-01',
#                                    '1970-01-01 23:59:00',
#                                    freq=interval)
#    Results= Results.set_index('Time')
#
# create a float version of the date index in prep for ploting
#    Results['FloatTime'] = mdates.date2num(Results.index)
#    return Results
#
# Histogram Log Returns ####################################
# def
# plt.figure()
# Price1['LogRet'].dropna().hist(normed=True, bins= 25, alpha=0.2)
# Price1['LogRet'].dropna().plot(kind='kde')
# show()
# To find outliers:
# s > s.std() * 2
# Price15['LogRet'].dropna().std() * 3
# Price15['LogRet'] > Price15['LogRet'].std() * 3
# I no longer think I have a problem with outliers.
# Create average day using average log return ##############
def AverageDay(ohlc_series):
    """
    Creates an average day from a OHLC series based on cumulative log
    returns.
    This won't work well on gapped data (where close
    of one candle isn't the same as the open of the next.)
    Includes metadata start/stop column in metaRange
    This works best with high frequency data (1min) I don't think much
    is gained
    by looking at higher time frames. The output column (CumLogRetBP)
    is multiplied
    by 10, 000 to get basis points which is close enough to a pip to be
    comparable.
    Note: Works faster if passed data without weekends and removed periods.
    # TO DO:
    # 0: Super slow
    # 1: Investigate the presence of gaps.
    # 2: Adapt to only include certain days. For example just look at last
    #    business day of month for fixings.
    # 3: Further divide days into up days and down days.
    # 3.5: Plot: odd days, even days, all days and then same for down days.
    #    If effect is actually persistant, peaks should not move.
    # 4: Compare to CorMaps.
    # 5: Coplot the hourly and/or the 15 min CorLag. (That would be super
        interesting.)
    # 6: Try de-trending the data and then doing this analysis on the
    residuals.
    # 7: Contains extremely inefficient df.set_value. Remove it if you can.
    # 8. the dataframe that gets looped through contains OHLC data as well
    # would it run faster if the df contained only log returns?
    """
    prices = ohlc_series.copy()
    # This first section is taken from CandleTrans
    start_time = time.time()
    tsInterval = prices.index[1] - prices.index[0]
    if tsInterval.seconds / 60 > 60:  # Error? .total_seconds()
        raise Exception("Time series frequency must be <= 60 minutes.")
    if 60 % (tsInterval.seconds / 60) != 0:  # Error? .total_seconds()
        raise Exception("Frequency must be 60 or integer factor of 60.")
    interval = tsInterval.seconds // 60  # Error? .total_seconds()
    # StrInterval = str(int(interval)) + 'Min'
    # Make simple returns.
    prices['SimpRet'] = (prices['Close'] / prices['Open']) - 1
    # Make: log returns (Be sure float division is working.)
    prices['LogRet'] = log(prices['Close'] / prices['Open'])
    time_diff = str(round(time.time() - start_time, 1))
    # Resample Section #####
    # prices = prices.resample(StrInterval)
    # Now take prices and do an average day with it.
    # This next section is taken from CandleStats.

    bins = 24 * (60 // interval)  # Keep integer division here.
    Results = pd.DataFrame(range(0, bins))
    # Create indices for later plotting
    intervalStr = str(int(interval)) + 'Min'
    Results['Time'] = pd.date_range('1970-01-01', '1970-01-01 23:59:00',
                                    freq=intervalStr)
    Results['FloatTime'] = mdates.date2num(Results.Time)
    # Write metadata:
    Results = Results.set_value(0, 'metaRange', prices.index[0])
    Results = Results.set_value(1, 'metaRange', prices.index[-1])
    # Some functions below don't tolerate NANs. Get rid of.
    prices = prices.dropna(axis=0)  # Don't drop NANs DF average won't work.
    # Initialize 3 target DF columns. Make sure they have the right dtypes.
    Results['LogRet'] = 0.1  # Force float64. Otherwise, will multi-copy df.
    Results['CumLogRetBP'] = 0.1
    # Now do main analysis loop
    for num in range(0, bins):
        # Select the data
        LoopPrices = prices[(((prices.index.hour * (60 / interval) +
                               (prices.index.minute // interval))) == (num))]
        Results.LogRet.ix[num] = average(LoopPrices.LogRet)
        progress = (100 * num // bins)
        sys.stdout.write("\rPercent complete: %d%%   " % (progress))
        sys.stdout.flush()
    time_diff = str(round(time.time() - start_time, 1))
    print '\rCompleted in %s seconds' % time_diff
    # Now make cumulative log returns. multiply by 10000 to get basis points.
    Results['CumLogRetBP'] = 10000 * cumsum(Results['LogRet'])
    del Results['LogRet']
    Results = Results.set_index('Time')
    return Results

# Do this after for DST=true, true?
# TrueDay['CumLogRetBP'] = TrueDay['CumLogRetBP'].shift(60)
# Can you convert to OHLC series?

# Now Make an average Hour ##################################


def AverageHour(prices):
    """
    Looks at an average hour. Works best when use on OHLC series that have
    been through classify_ohlc() and then have been filtered (for example)
    by
    hours that had a net gain:
    ohlc_series = ohlc_series[(ohlc_series.UpHour == True)]
    Or by further filtering to only look at hor between 7 and noon UTC:
    ohlc_series = ohlc_series[((ohlc_series.index.hour > 7) & \
        (ohlc_series.index.hour < 12))]
    Many combinations of filters are possible.

    Resample into 15 min candles to visulaize.
    """
    # Pushed to the bottom of my list.
    # An average hour actually isn't that interesting.

# Do 1D lag correlation #################################


def CorLag(*args, **kwargs):
    """
    Takes two pd.timeseries and returns lag correlation daily seasonality
    values. Both ts indexes must be MONONOTONIC and GAPLESS and be equal in
    frequency.
    Interval must be 60 or less that is an integer factor of 60.
    1, 2, 3, 4, 5, 6, 10...

    Leaving mask=True, values below 95%CL are set to zero.

    For lag=1, the value returned gives the correlation of ts1 with the
    previous
    period of ts2. If ts1==ts2 then it returns the autocorrelation of ts1
    with
    the previous time period. This is useful because a positive value for a
    period means that it's a period that will continue the behavior of the
    previous candle. I don't know if this is the way autocorrelation is done
    normally but it's most useful for prediction so I'm not going to change
    it.

    # Also, the value given corresponds to the y-axis (the present) of CorMap
    # and CorMapPlot.
    # TO DO:
    # 1: Create a metadata column that contains start and end time in rows [0]
    and [1].
    # 7: Contains extremely inefficient df.set_value. Remove it if you can.
    # 8: Metadata says the begin/end is the begin/end of ts1. This is okay if
    # ts1=ts1 but not if they are different. Fix it?
    # 9: make it so one ts becomes two. (Do this to all your correlation fns.
    # 10: Update docstring with kwargs.
    """
    start_time = time.time()
    # unpack *args
    if len(args) == 1:
        ts1 = args[0].copy()
        ts2 = args[0].copy()
    elif len(args) == 2:
        ts1 = args[0].copy()
        ts2 = args[1].copy()
    else:
        raise Exception("Too many time series arguments.")
    # Unpack **kwargs
    if 'method' in kwargs:
        method = kwargs.pop('method')
    else:
        method = 'pearson'
    if 'lag' in kwargs:
        lag = kwargs.pop('lag')
    else:
        lag = 1
    if 'mask' in kwargs:
        mask = kwargs.pop('mask')
    else:
        mask = False
    # Check data is good.
    ts1Interval = ts1.index[1] - ts1.index[0]
    ts2Interval = ts2.index[1] - ts2.index[0]
    if ts1Interval.seconds != ts2Interval.seconds:  # Error? .total_seconds()
        raise Exception("Time series don't have equal frequencies.")
    if ts1Interval.seconds / 60 > 60:  # Error? .total_seconds()
        raise Exception("Time series frequency must be <= 60 minutes.")
    if 60 % (ts1Interval.seconds / 60) != 0:  # Error? .total_seconds()
        raise Exception("Frequency must be 60 or integer factor of 60.")
    if ((len(ts1.shape) > 1) | (len(ts1.shape) > 1)):
        raise Exception("Times series must have only one column.")
    # Make an appropriate time index to go with it.
    interval = ts1Interval.seconds // 60  # Error? .total_seconds()
    bins = 24 * (60 // interval)  # Keep integer division here.
    intervalStr = str(int(interval)) + 'Min'
    Results = pd.DataFrame(range(0, bins))
    Results['Time'] = pd.date_range('1970-01-01', '1970-01-01 23:59:00',
                                    freq=intervalStr)
    Results['FloatTime'] = mdates.date2num(Results.Time)
    # Write date range metadata
    Results = Results.set_value(0, 'metaRange', ts1.index[0])
    Results = Results.set_value(1, 'metaRange', ts1.index[-1])

    step = 1
    total = bins
    for num in range(0, bins):
        progress = ((step / total) * 100) // 1
        sys.stdout.write("\rPercent complete: %d%%   " % (progress))
        sys.stdout.flush()
        step += 1
        RowTs1 = ts1[((ts1.index.hour * (60 / interval) +
                      (ts1.index.minute // interval))) == (num)]
        LaggedTs2 = ts2.shift(lag)
        ColTs2 = LaggedTs2[((LaggedTs2.index.hour * (60 / interval) +
                            (LaggedTs2.index.minute // interval))) == (num)]

        Correlation = RowTs1.corr(ColTs2, method=method, min_periods=10)
        if math.isnan(Correlation):
                Correlation = 0
        Results = Results.set_value(num, 'LagCor', Correlation)
        Intersect = RowTs1 + ColTs2
        Intersect = len(Intersect.dropna())
        Results = Results.set_value(num, 'CritVal95',
                                   (CriticalVal(Intersect,
                                    cl="95",
                                    method=method)))
        Results = Results.set_value(num, 'CritVal99',
                                   (CriticalVal(Intersect,
                                    cl="99",
                                    method=method)))
        if mask:
            print Intersect, Correlation, (CriticalVal(Intersect))
            # Show intersection of two time series.
            # print len(RowTs1), len(ColTs2), Intersect ,
            # print CriticalVal(Intersect), Correlation
            if (fabs(Correlation) < fabs(CriticalVal(Intersect))):
                Results = Results.set_value(num, 'LagCor', 0)
    time_diff = str(round(time.time() - start_time, 1))
    print '\rCorrelation sequence completed in %s seconds' % time_diff
    return Results

# CorLagPlot  #########################################


def CorLagPlot(LagSeries):
    """
    Plots 1D correaltion plot generated from CorLag.
    TO DO:
    Metadata propagation. Chart should be labeled.
    """
    # First call the main window
    fig1 = plt.figure(figsize=(8, 6))  # x by y
    # Now define the first (or only) subplot:
    ax1 = fig1.add_subplot(111)  # 111: 1x1 grid, first subplot"
    # Do this to plot: (you can repeat the command for more series
    # ax1.plot_date(LagSeries['FloatTime'], LagSeries['LagCor'],
    #         marker=None, linestyle='-', linewidth=0.5, color='r',
    #         xdate=True, ydate=False)
    # Below I'm using where='pre'. Not quite right but 'mid' isn't either.
    ax1.step(LagSeries['FloatTime'], LagSeries['CritVal95'],
             where='pre', linewidth=1, color='y')
    ax1.step(LagSeries['FloatTime'], -1 * LagSeries['CritVal95'],
             where='pre', linewidth=1, color='y')
    ax1.step(LagSeries['FloatTime'], LagSeries['CritVal99'],
             where='pre', linewidth=1, color='g')
    ax1.step(LagSeries['FloatTime'], -1 * LagSeries['CritVal99'],
             where='pre', linewidth=1, color='g')
    ax1.step(LagSeries['FloatTime'], LagSeries['LagCor'],
             where='pre', linewidth=1, color='r')
    # Set the x limits to be an entire day so the stripes look right.
    xlimits = ax1.get_xlim()
    ax1.set_xlim(xlimits[0], xlimits[0] + 1)

    def Zebra(target, ZebraNum):
        xlimits = target.get_xlim()
        xStart = xlimits[0]
        xEnd = xlimits[1]
        ZebraNum = ZebraNum / 2
        xLength = xEnd - xStart
        ZebraWidth = (xLength / ZebraNum) / 2
        i = xStart
        # i = i + (xLength/ZebraNum) # Uncomment for even shading.
        while i < xEnd:
            target.axvspan(i, i + ZebraWidth, facecolor='0.2', alpha=0.1)
            i = i + (xLength / ZebraNum)
        return target
    Zebra(ax1, 24)
    # format the coords message box
    ax1.format_xdata = mdates.DateFormatter('%H:%M')
    # ax1.format_ydata = price
    # fig1.xticks(rotation=30) doesn't work
    ax1.xaxis_date()
    fig1.autofmt_xdate()
    ax1.grid(True)
    # Add some labels:
    xlabel('Candle ending time: UTC', {'fontsize': 18})  # Global?
    ylabel('Correlation coefficient', {'fontsize': 18})  # Global?
    plt.title('Daily Seasonality')  # This might be global
    show()

# Lookup Pearson coefficient Critical Values ###################


def CriticalVal(n, cl='95', method='pearson'):
    """
    Given a sample size, returns 2-tailed Pearson correlation critical
    value
    from table. Takes values from 1-1000.
    Needs: 'from scipy.interpolate import interp1d' .
    For one-tailed probabilities, half the p value. For example, the 0.05
    table
    becomes a 0.025 table for one tailed and the 0.01 table becomes 0.005.
    This table is cloes enough to use for spearman's rho as well, but not
    kendall
    tau.
    """
    DF = n - 2
    if (method == 'kendall'):
        raise Exception("Kendall tau not yet implemented.")
    if cl == '95':
        ValTable95 = np.array([[1, 0.99692], [2, 0.950], [3, 0.878],
                               [4, 0.811], [5, 0.755], [6, 0.707],
                               [7, 0.666], [8, 0.632], [9, 0.602],
                               [10, 0.576], [11, 0.553], [12, 0.532],
                               [13, 0.514], [14, 0.497], [15, 0.482],
                               [16, 0.468], [17, 0.456], [18, 0.444],
                               [19, 0.433], [20, 0.423], [21, 0.413],
                               [22, 0.404], [23, 0.396], [24, 0.388],
                               [25, 0.381], [26, 0.374], [27, 0.367],
                               [28, 0.361], [29, 0.355], [30, 0.349],
                               [35, 0.325], [40, 0.304], [45, 0.288],
                               [50, 0.273], [60, 0.250], [70, 0.232],
                               [80, 0.217], [90, 0.205], [100, 0.195],
                               [120, 0.178], [150, 0.159], [300, 0.113],
                               [500, 0.086], [1000, 0.062]])
        lookup = interp1d(ValTable95[:, 0], ValTable95[:, 1], kind='cubic')
    elif cl == '99':
        ValTable99 = np.array([[1, 0.999877], [2, 0.990], [3, 0.959],
                               [4, 0.917], [5, 0.874], [6, 0.834],
                               [7, 0.798], [8, 0.765], [9, 0.735],
                               [10, 0.708], [11, 0.684], [12, 0.661],
                               [13, 0.641], [14, 0.623], [15, 0.606],
                               [16, 0.590], [17, 0.575], [18, 0.561],
                               [19, 0.549], [20, 0.537], [21, 0.526],
                               [22, 0.515], [23, 0.505], [24, 0.496],
                               [25, 0.487], [26, 0.478], [27, 0.470],
                               [28, 0.463], [29, 0.456], [30, 0.449],
                               [35, 0.418], [40, 0.393], [45, 0.372],
                               [50, 0.354], [60, 0.325], [70, 0.302],
                               [80, 0.283], [90, 0.267], [100, 0.254],
                               [120, 0.232], [150, 0.208], [200, 0.181],
                               [300, 0.149], [400, 0.129], [500, 0.115],
                               [1000, 0.081]])
        lookup = interp1d(ValTable99[:, 0], ValTable99[:, 1], kind='cubic')
    else:
        raise Exception("CL not available. Use cl='99' or cl='95'")
    if DF <= 1000:
        val = float(lookup(DF))
    elif DF > 1000:
        val = 0.062
    elif DF < 1:
        raise Exception("Critical values not defined below 1.")
    return val


# Make 2D Correlation Map #################################
def CorMap(*args, **kwargs):
    """
    Takes one or two pd.timeseries and returns a lag correlation (or
    autocorrelation) daily seasonality heatmap. Both ts indexes must be
    monotonic and gapless and be equal in frequency.
    Interval must be 60 or less that is an integer factor of 60.
    1, 2, 3, 4, 5, 6, 10...
    Leaving mask=True, values below 95%CL are set to zero.
    Setting mask=False make it run in 75% of the time as with mask=True
    Another advantage of spearman's rho is that the critical value table is
    very close to pearson's, especially for values over 20. For Kendall, look
    it
    up in Sheskin's Handbook of Parametric and Nonparametric statstical
    procedures: pg 1174 of 1184.
    Uses pearson correaltion by default which is only valid with normal
    distributions (when the sample size is small). It is also quite sensitive
    to outliers. Use kendall or spearman if these things are a concern or
    if the data are not linearly related. Spearman's rho might be the most
    closely
    related to pearsons r but it's not considered to be quanitative. Kendalls
    tau represents actual probabilitys. Kendalls Tau represents a probability,
    i.e., the difference between the probability that the observed data are in
    the same order versus the probability that the observed data are not in
    the
    same order.
    TO DO:
    1: Count total point exceeding critical value and print it, and it as a
    percent
    of total possible. This should give you an indicaiton of how accurate your
    critical values are. e.g. You shoulld get 5% points of alpha=0.5.
    2: Create a metadata column that contains start and end time in rows [0]
    and [1].
    3: Double check that lags stay within one day. lags should not cross
    midnight.
    I think I did it this way. I checked once but should check again.
    4: update docstring with kwargs
    """
    start_time = time.time()
    # Unpack args
    if len(args) == 1:
        ts1 = args[0].copy()
        ts2 = args[0].copy()
    elif len(args) == 2:
        ts1 = args[0].copy()
        ts2 = args[1].copy()
    else:
        raise Exception("Too many time series arguments.")

    # Unpack kwargs
    if 'method' in kwargs:
        method = kwargs.pop('method')
    else:
        method = 'pearson'
    if 'cl' in kwargs:
        cl = kwargs.pop('cl')
    else:
        cl = '95'
    if 'mask' in kwargs:
        mask = kwargs.pop('mask')
    else:
        mask = True

    # Check if data is good:
    ts1Interval = ts1.index[1] - ts1.index[0]
    ts2Interval = ts2.index[1] - ts2.index[0]
    if ts1Interval.seconds != ts2Interval.seconds:  # Error? .total_seconds()
        raise Exception("Time series don't have equal frequencies.")
    if ts1Interval.seconds / 60 > 60:  # Error? .total_seconds()
        raise Exception("Time series frequency must be <= 60 minutes.")
    if 60 % (ts1Interval.seconds / 60) != 0:  # Error? .total_seconds()
        raise Exception("Frequency must be 60 or integer factor of 60.")
    if ((len(ts1.shape) > 1) | (len(ts1.shape) > 1)):
        raise Exception("Times series must have only one column.")
    interval = ts1Interval.seconds // 60  # Error? .total_seconds()
    bins = 24 * (60 // interval)  # Keep integer division here.
    map = np.zeros(shape=(bins, bins))  # Square array for data to go into.
    step = 1
    total = bins / 2 + (bins * bins) / 2
    # Do Raster Loop that stops at the diagonal.
    for row in range(bins):
        # zero = midnight, bins = midnight. loops goes 0->(bins-1)
        RowTs1 = ts1[((ts1.index.hour * (60 / interval) +
                      (ts1.index.minute // interval))) == (row)]
        for col in range(bins):
            if col > row:
                break
            # print(row, col, (row-col))
            progress = ((step / total) * 100) // 1
            sys.stdout.write("\rPercent complete: %d%%   " % (progress))
            sys.stdout.flush()
            step += 1
            TempTs2 = ts2.shift(row - col)
            ColTs2 = TempTs2[((TempTs2.index.hour * (60 / interval) +
                              (TempTs2.index.minute // interval))) == (row)]
            # Now set the values
            Correlation = RowTs1.corr(ColTs2, method=method, min_periods=10)
            map[row, col] = Correlation
            if math.isnan(Correlation):
                Correlation = 0
            if mask:
                # Count intersection of two time series.
                Intersect = RowTs1 + ColTs2
                Intersect = len(Intersect.dropna())
                # print len(RowTs1), len(ColTs2), Intersect
                # Try gapped and ungapped data.
                # print CriticalVal(Intersect), Correlation
                critical = CriticalVal(Intersect, cl=cl, method=method)
                critcheck = (fabs(Correlation) < fabs(critical))
                if critcheck:
                    map[row, col] = 0
    time_diff = str(round(time.time() - start_time, 1))
    print '\rCorrelation map completed in %s seconds' % time_diff
    return map

# Plot 2D correlation ##################################


def CorMapPlot(Map, scale='relative'):
    """
    Plots daily seasonality heatmap of correlation data created by CorMap.
    It doesn't like Nans.

    TO DO:
    In conjunction with CorMap, define 2 more layers: One each for 95% and
    99%
    masks. Because of the time involved in making a CorMap, the Masking
    funtionality should be moved here instead of in CorMap.

    """
    # Uses imshow instead
    # of pcolormesh because pcolormesh won't work with masked arrays.
    # Also, imshow
    # makes grid coorinates at the center of each spot rather than
    # the lower left
    # hand corner. Use a color map that has white in the middle: BrBG,
    # PRGn, PiYG, PuOr, RdBu, RdGy. The last two are nice.
    if Map.shape[0] != Map.shape[1]:
        raise Exception("Array is not square.")
    interval = (24 / Map.shape[0]) * 60
    bins = 24 * (60 // interval)  # Keep integer division here. _future_
    mask = np.diag(np.ones(bins))
    masked_Map = ma.masked_array(Map, mask)
    cm.RdBu.set_bad(color='black', alpha=None)  # Mask color.
    if scale == 'absolute':
        colormin = -1
        colormax = 1
    elif scale == 'relative':
        Mapmin = fabs(ma.min(masked_Map))
        Mapmax = fabs(ma.max(masked_Map))
        limit = max(Mapmin, Mapmax)
        colormin = -1 * limit
        colormax = limit
    else:
        raise Exception("Unknown scale argument.")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    i = ax.imshow(masked_Map,
                  interpolation="nearest",
                  cmap=cm.RdBu,
                  vmin=colormin,
                  vmax=colormax)
    fig.colorbar(i)
    # When calling, do i = ax.imshow();
    # fig.colorbar(i, cax='colorbar_ax')
    # Now re-assign ax.format_coord so plot is interactive.
    numrows, numcols = masked_Map.shape

    def format_coord(x, y):
        col = int(x + 0.5)
        colH = num2date(1 + col / bins).hour
        colM = num2date(1 + col / bins).minute
        row = int(y + 0.5)
        rowH = num2date(1 + row / bins).hour
        rowM = num2date(1 + row / bins).minute
        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            z = masked_Map[row, col]
            return 'x(if)=%02d:%02d, '\
                   'y(then)=%02d:%02d, '\
                   'Correlation=%1.2f' % (colH, colM, rowH, rowM, z)
        else:
            return 'x=%1.4f, y=%1.4f' % (x, y)
    # Dynamically label the plot.
    intervalString = str(int(interval)) + 'Min'
    annotation = 'Max positive corr.:%1.2f \n'\
                 'Max negative corr.:%1.2f\n'\
                 'Color scale = %s\n'\
                 'Tick size = %s' % (Mapmax, (-1 * Mapmin),
                                     scale, intervalString)
    text(0.5, 0.85, annotation,
         horizontalalignment='left',
         verticalalignment='center',
         family='serif',
         transform=ax.transAxes)
    # Add some labels:
    xlabel('Past', {'fontsize': 12})
    ylabel('Present', {'fontsize': 12})
    plt.title('Correlation Seasonality')

    ax.format_coord = format_coord
    show()

if __name__ == '__main__':
    import sys
    sys.exit(1)
