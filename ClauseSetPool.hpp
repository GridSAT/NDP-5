// ClauseSetPool.hpp  
//
// Clause Set Pool for NDP-5.6.7
//
// Copyright (c) 2025 GridSAT Stiftung
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
// 
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
// 
// GridSAT Stiftung - Georgstr. 11 - 30159 Hannover - Germany - ipns://gridsat.eth - info@gridsat.io
//
//
// +++ READ.me +++
// 
// Save to working directory of NDP-5.6.7
//   
#ifndef CLAUSE_SET_POOL_HPP
#define CLAUSE_SET_POOL_HPP

#include <vector>
#include <cstdint>



// Using default allocation for ClauseSet.

struct Clause3 {
    int l[3];
};

using ClauseSet = std::vector<Clause3>;

struct ClauseSetPool {
        std::vector<ClauseSet*> freeList;
        
        ClauseSet* obtain(std::size_t reserveSize = 0) {
                if (!freeList.empty()) {
                    ClauseSet* cs = freeList.back();
                    freeList.pop_back();
                    cs->clear();
                    if(reserveSize)
                            cs->reserve(reserveSize);
                    return cs;
                }
                ClauseSet* cs = new ClauseSet();
                if(reserveSize)
                        cs->reserve(reserveSize);
                return cs;
        }
        
        void release(ClauseSet* cs) {
                freeList.push_back(cs);
        }
        
        ~ClauseSetPool() {
                for(auto cs : freeList)
                        delete cs;
        }
};

#endif // CLAUSE_SET_POOL_HPP