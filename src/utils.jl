function set_offset!(offset::Array{Int,2})
    N = size(offset,1)
    idx = 1
    if N == 1
        for s1 in (0,1)
            offset[1,idx] = s1
            idx += 1
        end
    elseif N ==2
        for s1 in (0,1)
            for s2 in (0,1)
                offset[1,idx] = s1
                offset[2,idx] = s2
                idx += 1
            end
        end
    else
        for s1 in (0,1)
            for s2 in (0,1)
                for s3 in (0,1)
                    offset[1,idx] = s1
                    offset[2,idx] = s2
                    offset[3,idx] = s3
                    idx += 1
                end
            end
        end
    end
end