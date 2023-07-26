#=
Algorithm for finding bridged electrodes translated from MNE-Python 
with some modifications based on the original MATLAB implementation from:
Alschuler, D. M., Tenke, C. E., Bruder, G. E., & Kayser, J. (2014). Identifying electrode 
bridging from electrical distance distributions: a survey of publicly-available EEG data 
using a new method. Clinical neurophysiology, 125(3), 484–490. 
https://doi.org/10.1016/j.clinph.2013.08.024

Initial implementation written during Brainhack Warsaw 2023 by:
- Ibrahim Vefa Arslan
- Katarzyna Jałocha
- Dominik Wiącek
=#
function find_bridges(data, srate; timeSpan=2., cutoff=16, threshold=0.5, scaling=false)
    # Estimate the number of segments from data
    nSamples, nChannels = size(data)
    sampleSpan = round(Int, timeSpan * srate)
    segIdx = 1:sampleSpan:(nSamples-sampleSpan)
    nSegments = length(segIdx)

    # Container for electrical distance estimations
    electricalDistance = fill(NaN, (nChannels, nChannels, nSegments))
    buffer = zeros(sampleSpan)

    for idx in 1:nSegments
        @views electrical_distance!(data, segIdx[idx], sampleSpan, nChannels, electricalDistance[:,:,idx], buffer)
    end

    # Scaling factor applied in the original implementation
    if scaling
        electricalDistance .*= (100/median(electricalDistance[electricalDistance .> 0]))
    end

    # Check if there are enough of pairs of signals to make calculations worthwhile
    if sum(electricalDistance .< cutoff) / nSegments < threshold
        println("No bridged electrodes found.")
        return electricalDistance
    end

    # Estimate the distibution of electrical distances below the local minimum cutoff
    distribution = kde(electricalDistance[electricalDistance .< cutoff])
    # Find the actual local minimum
    # eBridge just looks for a minimum in range, MNE-Python minimizes the density function
    # In Julia we could do such a minimization e.g.:
    # using Optim
    # funkde(x) = pdf(density, x)
    # localMin = optimize(funkde, 0, lmCutoff).minimizer
    searchRegion = 0 .< distribution.x .< cutoff
    minVal, searchIdx = findmin(identity, distribution.density[searchRegion])
    minIdx = distribution.x[searchRegion][searchIdx]
    println("Local minimum found at $minIdx")

    # Allocate the result boolean matrix indicating pairs of bridged electrodes
    bridgedMatrix = zeros(Bool, (nChannels, nChannels))
    for chan in 1:nChannels
        for chan2 in chan+1:nChannels
            # Mark channel pairs that have more segments with distance below local minimum
            # than specified by the threshold.
            @views bridgedMatrix[chan, chan2] = sum(electricalDistance[chan, chan2, :] .< minIdx)/nSegments > threshold
        end
    end

    bridgedPairs = sort(Tuple.(findall(x -> x, bridgedMatrix)))
    println("Bridge identified between $(length(bridgedPairs)) electrode pairs.")
    return bridgedPairs, bridgedMatrix, electricalDistance
end

function electrical_distance!(data, segID, nSamples, nChannels, electricalDistance, buffer)
    for chan in 1:nChannels
        @views for chan2 in chan+1:nChannels
            buffer .= data[segID:segID+nSamples-1, chan] .- data[segID:segID+nSamples-1, chan2]
            electricalDistance[chan, chan2] = var(buffer, corrected=false)
        end
    end
end