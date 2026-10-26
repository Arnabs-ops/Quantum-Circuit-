"""Four-address signed-gauge ANGLE XOR compiler with measured bank erasure.

SPDX-License-Identifier: Apache-2.0

Reviewed port of notebook #07, ahmedberatozer/quantum-ai-df-thc-group-compression,
internal Version 2, cells 8 (gauge_qrom.py) and 3 (banked_lab.py), supplied
2026-10-24 snapshot; own Kaggle version ID unknown. Apache-2.0 code licence
recorded in public/READING.md. Retained lineage: Kozykorpesh / Sumit Chongder
banked QROM, through Kirill Tym's quantum-circuits-qrom-tower42 v10 and Artem
(anazemcev)'s truncated-ANF and split-ANF eraser. Public source was read as
untrusted text, never imported or executed. No numerical checkpoint is copied.

All public targets may be arbitrary. Only the private bank is initialized
clean. Every illegal high-address value has the same polynomial completion
in initialization and measured erasure, so cleanup is not dependent on an
unstated zero-padding assumption.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'data/comp/kickmix'))
import kickmix as km


def mobius(values):
    out=list(map(int,values))
    if not out or len(out)&(len(out)-1):
        raise ValueError('GF(2) transform length must be a power of two')
    for bit in range((len(out)-1).bit_length()):
        step=1<<bit
        for lo in range(0,len(out),2*step):
            for offset in range(step):
                out[lo+step+offset]^=out[lo+offset]
    return out


def bits(value):
    while value:
        one=value&-value
        yield one.bit_length()-1
        value^=one


def xor_product(builder,left,right,targets):
    """One CCX fans out a product into arbitrary targets, with no clean pivot."""
    targets=list(targets)
    if not targets:return
    pivot=targets[0]
    for target in targets[1:]:builder.cx(pivot,target)
    builder.ccx(left,right,pivot)
    for target in reversed(targets[1:]):builder.cx(pivot,target)


def initialize_forest(builder,table,address,outputs,scratch):
    """Simple fixed forest when scratch holds every non-leaf prefix product."""
    address,outputs,scratch=map(list,(address,outputs,scratch))
    count,n=len(table),len(address)
    coeff=mobius(list(table)+[0]*((1<<n)-count))
    coeff[count:]=[0]*((1<<n)-count)
    completed=mobius(coeff)
    if completed[:count]!=list(table):raise AssertionError('prefix completion changed a legal row')
    stats=dict(interior_products=0,recomputed_products=0,pebble_sites=0,
               terminal_products=0,scratch_qubits=len(scratch),
               high_rows=count,high_address_bits=n,mode='fixed forest')

    def children(mask):
        return [(mask|(1<<bit),bit) for bit in range(mask.bit_length(),n)
                if (mask|(1<<bit))<count]

    def targets(mask):return [outputs[bit] for bit in bits(coeff[mask])]

    def walk(mask,control):
        ts=targets(mask)
        if ts:
            if control is None:builder.x(km.array(ts))
            else:builder.cx(control,km.array(ts))
        for child,bit in children(mask):
            if mask==0:walk(child,address[bit])
            elif not children(child):
                ts=targets(child)
                if ts:
                    xor_product(builder,control,address[bit],ts)
                    stats['terminal_products']+=1
            else:
                product=scratch[child.bit_count()-2]
                builder.init_and(control,address[bit],product)
                stats['interior_products']+=1
                walk(child,product)
                builder.del_and(control,address[bit],product)

    walk(0,None)
    return completed,stats


def initialize_pebbled(builder,table,address,outputs,scratch):
    """Prefix ANF forest, terminal target leaves and safe ancestor pebbling."""
    address,outputs,scratch=map(list,(address,outputs,scratch))
    count,n=len(table),len(address)
    if not 1<count<=1<<n or count<=1<<(n-1):
        raise ValueError('table must be a prefix using every high-address bit')
    coeff=mobius(list(table)+[0]*((1<<n)-count))
    coeff[count:]=[0]*((1<<n)-count)
    completed=mobius(coeff)
    if completed[:count]!=list(table):
        raise AssertionError('triangular completion changed a legal row')
    stats=dict(interior_products=0,recomputed_products=0,pebble_sites=0,
               terminal_products=0,scratch_qubits=len(scratch),
               high_rows=count,high_address_bits=n,mode='ancestor pebbled')

    @lru_cache(None)
    def children(mask):
        return tuple((mask|(1<<bit),bit) for bit in range(mask.bit_length(),n)
                     if (mask|(1<<bit))<count)

    @lru_cache(None)
    def needed(mask):
        return max([1+needed(child) for child,_ in children(mask) if children(child)]+[0])

    def targets(mask):return [outputs[bit] for bit in bits(coeff[mask])]

    def walk(mask,control,path,free):
        erased=[]
        if mask.bit_count()==5 and needed(mask)>len(free):
            shortage=needed(mask)-len(free)
            if not 0<shortage<=len(path)-1:
                raise ValueError('five-scratch pebbling schedule cannot cover this table')
            erased=path[:shortage]
            for left,right,target in reversed(erased):
                builder.del_and(left,right,target)
                free.append(target)
            stats['pebble_sites']+=1
        ts=targets(mask)
        if ts:
            if control is None:builder.x(km.array(ts))
            else:builder.cx(control,km.array(ts))
        for child,bit in children(mask):
            if mask==0:
                walk(child,address[bit],path,free)
            elif not children(child):
                ts=targets(child)
                if ts:
                    xor_product(builder,control,address[bit],ts)
                    stats['terminal_products']+=1
            else:
                if not free:raise ValueError('scratch exhausted in prefix forest')
                target=free.pop()
                builder.init_and(control,address[bit],target)
                stats['interior_products']+=1
                walk(child,target,path+[(control,address[bit],target)],free)
                builder.del_and(control,address[bit],target)
                free.append(target)
        for left,right,target in erased:
            free.remove(target)
            builder.init_and(left,right,target)
            stats['recomputed_products']+=1

    walk(0,None,[],scratch.copy())
    return completed,stats


def erase_lookup(builder,table,address,target,low_bits=5):
    """X-measure a clean lookup bank and exactly cancel its measurement phase.

    Reset bank qubits are reused for low-half monomials and high-half carries.
    Their own del_and calls correct all newly introduced HMR phases.
    """
    address,target=map(list,(address,target))
    n=len(address)
    if len(table)!=1<<n or not 0<low_bits<n:
        raise ValueError('eraser needs a full table and two nonempty address halves')
    if any(value<0 or value>=1<<len(target) for value in table):
        raise ValueError('lookup value exceeds private bank width')
    coeff=mobius(table)
    if mobius(coeff)!=list(table):raise AssertionError('eraser ANF identity failed')
    measurements=builder.alloc_bits(len(target))
    builder.hmr(km.array(target),measurements)
    parity=builder.alloc_bits(1)[0]
    high_bits=n-low_bits
    low_masks=[mask for mask in range(1,1<<low_bits) if mask.bit_count()>1]
    needed=len(low_masks)+high_bits-1
    extra=builder.alloc_qubits(max(0,needed-len(target)))
    workspace=target+list(extra)
    low={1<<bit:address[bit] for bit in range(low_bits)}
    for mask,qubit in zip(low_masks,workspace):
        last=mask.bit_length()-1
        builder.init_and(low[mask^(1<<last)],address[last],qubit)
        low[mask]=qubit
    carries=workspace[len(low_masks):needed]

    def correct(high_mask,high_qubit):
        for low_mask in range(1<<low_bits):
            coefficient=coeff[low_mask|(high_mask<<low_bits)]
            if not coefficient:continue
            builder.bit_store(parity,False)
            for bit in bits(coefficient):builder.cx(measurements[bit],parity)
            if not low_mask and not high_mask:
                builder.neg_if(parity)
            else:
                builder.push_condition(parity)
                if not low_mask:builder.z(high_qubit)
                elif not high_mask:builder.z(low[low_mask])
                else:builder.cz(low[low_mask],high_qubit)
                builder.pop_condition()

    def walk(mask,control):
        correct(mask,control)
        for bit in range(mask.bit_length(),high_bits):
            child=mask|(1<<bit)
            if not mask:walk(child,address[low_bits+bit])
            else:
                product=carries[mask.bit_count()-1]
                builder.init_and(control,address[low_bits+bit],product)
                walk(child,product)
                builder.del_and(control,address[low_bits+bit],product)

    walk(0,None)
    for mask in reversed(low_masks):
        last=mask.bit_length()-1
        builder.del_and(low[mask^(1<<last)],address[last],low[mask])
    builder.free(extra)
    return dict(high_address_bits=n,low_bits=low_bits,other_bits=high_bits,
                retained_products=len(low_masks),traversal_carries=high_bits-1,
                scratch_qubits=needed,extra_qubits=len(extra),
                toffoli=(1<<low_bits)-low_bits-1+(1<<high_bits)-high_bits-1)


def tables(delta):
    """Return full base, three compact differences, and exact four-case certificate."""
    delta=np.asarray(delta)
    if delta.ndim!=2 or delta.shape[1]!=3 or len(delta)<8 or len(delta)%4:
        raise ValueError('four-address compiler requires (A,3), A>=8 divisible by four')
    if not np.issubdtype(delta.dtype,np.integer) or np.any(delta<0) or np.any(delta>=1024):
        raise ValueError('delta words must be ten-bit integers')
    if np.any(((delta>>9)^(delta>>8))&1):
        raise ValueError('ten-bit signed half-open gauge requires bit9=bit8')
    packed=[sum(int(value)<<(10*j) for j,value in enumerate(row)) for row in delta]
    compact=lambda value:sum(((value>>(10*j))&511)<<(9*j) for j in range(3))
    expand=lambda value:sum((((value>>(9*j))&511)|(((value>>(9*j+8))&1)<<9))<<(10*j) for j in range(3))
    base=[];diff=[[],[],[]]
    for lo in range(0,len(packed),4):
        a,b,c,d=packed[lo:lo+4]
        base.append(a)
        diff[0].append(compact(a^b))
        diff[1].append(compact(a^c))
        diff[2].append(compact(a^b^c^d))
    restored=[base[h]^expand((diff[0][h] if case&1 else 0)^
                            (diff[1][h] if case&2 else 0)^
                            (diff[2][h] if case==3 else 0))
              for h in range(len(base)) for case in range(4)]
    if restored!=packed:raise AssertionError('four-address integer certificate failed')
    values=[base[h]|(diff[0][h]<<30)|(diff[1][h]<<57)|(diff[2][h]<<84)
            for h in range(len(base))]
    return values,expand


def build(delta,brot=10,*,return_report=False,scratch_qubits=None):
    if brot!=10:raise ValueError('four-address compiler requires brot10')
    delta=np.asarray(delta)
    values,_=tables(delta)
    n=(len(delta)-1).bit_length()
    if n-2<=5:raise ValueError('this eraser requires more than five high-address bits')
    if scratch_qubits is None:scratch_qubits=6 if n+30+81+6<=128 else 5
    if scratch_qubits not in (5,6) or n+30+81+scratch_qubits>128:
        raise ValueError('scratch choice does not fit the128-qubit circuit cap')
    builder=km.CircuitBuilder()
    selection=builder.create_quantum_register(n,'selection')
    output=[qubit for j in range(3) for qubit in builder.create_quantum_register(10,f'word_{j}')]
    bank=list(builder.alloc_qubits(81))
    scratch=builder.alloc_qubits(scratch_qubits)
    high_bits=n-2
    need=max((mask.bit_count()-1 for mask in range(1,len(values))
              if any((mask|(1<<bit))<len(values) for bit in range(mask.bit_length(),high_bits))),default=0)
    initializer=initialize_forest if need<=scratch_qubits else initialize_pebbled
    completed,init=initializer(builder,values,selection[2:],output+bank,scratch)
    builder.free(scratch)
    both=builder.alloc_qubits(1)[0]
    builder.init_and(selection[0],selection[1],both)
    for s,control in enumerate((selection[0],selection[1],both)):
        for word in range(3):
            for bit in range(9):
                targets=[output[word*10+bit]]
                if bit==8:targets.append(output[word*10+9])
                xor_product(builder,control,bank[s*27+word*9+bit],targets)
    builder.del_and(selection[0],selection[1],both)
    builder.free(both)
    erasure=erase_lookup(builder,[value>>30 for value in completed],selection[2:],bank,low_bits=5)
    builder.free(km.array(bank))
    circuit=builder.finish_circuit()
    expected=init['interior_products']+init['recomputed_products']+init['terminal_products']+82+erasure['toffoli']
    actual=int(circuit.operation_statistics().num_toffoli_like)
    if actual!=expected:raise AssertionError('native circuit and derived operation counts differ')
    if circuit.num_qubits>128:raise ValueError('four-address circuit exceeds128 qubits')
    report=dict(rows=len(delta),brot=10,J=3,selection_bits=n,num_qubits=circuit.num_qubits,
                bank_qubits=81,initialization_scratch=scratch_qubits,initialization=init,erasure=erasure,
                controlled_difference_products=81,low_address_product=1,
                toffoli_per_call=actual,integer_table_identity='PASS')
    return (circuit,report) if return_report else circuit
