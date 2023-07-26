module Anomaly

import Statistics: var, median
import KernelDensity: kde, pdf

include("channels/amplitude.jl")
export peak2peak

include("channels/bridge.jl")
export find_bridges

end # module Anomaly
