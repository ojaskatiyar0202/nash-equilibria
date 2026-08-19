#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 19:37:38 2026

@author: Ojas
"""

import numpy as np
from itertools import permutations
from itertools import combinations


class ProductTwoActionGame:
    def __init__(self, n):
        """Initialize game with n players"""
        self.n = n
        self.A = self._generate_coefficients()
    
    def _generate_coefficients(self):
        A = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                A[i, j] = ((j - i) % self.n) / self.n
        return np.round(A, 2)
    
    def get_derangements_with_fixed_points(self, num_fixed):
        """
        Get all permutations with exactly num_fixed fixed points.
        
        Args:
            num_fixed: number of fixed points (0 to n)
        
        Returns:
            list of permutation tuples
        """
        from itertools import combinations
        elements = tuple(range(1, self.n + 1))
        results = []
        
        # Choose which positions are fixed
        for fixed_positions in combinations(range(self.n), num_fixed):
            # Get the non-fixed positions
            non_fixed_positions = [i for i in range(self.n) if i not in fixed_positions]
            non_fixed_elements = [elements[i] for i in non_fixed_positions]
            
            # Generate derangements of the non-fixed elements
            for perm in permutations(non_fixed_elements):
                # Check if it's a derangement (no element in its original position)
                is_derangement = True
                for i, pos in enumerate(non_fixed_positions):
                    if perm[i] == elements[pos]:
                        is_derangement = False
                        break
                
                if is_derangement:
                    # Build the full permutation
                    full_perm = [0] * self.n
                    # Place fixed elements
                    for pos in fixed_positions:
                        full_perm[pos] = elements[pos]
                    # Place deranged elements
                    for i, pos in enumerate(non_fixed_positions):
                        full_perm[pos] = perm[i]
                    
                    results.append(tuple(full_perm))
        
        return results
    
    def game_matrix(self, x):
        """Create M[i,j] = x_j - a^i_j with diagonal 0"""
        X_matrix = np.tile(x, (self.n, 1))
        M = X_matrix - self.A
        np.fill_diagonal(M, 0)
        return M
    
    def parity(self, M):
        """Convert to binary: 0 if M≥0, 1 if M<0"""
        return (M < 0).astype(int)
    
    def extract_coefficients(self, pi):
       """
       Extract coefficients a^{π(j)}_j for each column j.
       
       Args:
           pi: permutation tuple (1-indexed), e.g., (1, 3, 4, 2, 5)
       
       Returns:
           numpy array of coefficients [a^{π(1)}_1, a^{π(2)}_2, ..., a^{π(n)}_n]
       """
       # Convert to 0-indexed for array access
       row_indices = [p - 1 for p in pi]
       col_indices = list(range(len(pi)))
       
       return self.A[row_indices, col_indices]
   
    def possible_equilibria(self):
        """Get all possible permutations from 0 to n fixed points"""
        all_perms = []
        for num_fixed in range(self.n + 1):
            perms = self.get_derangements_with_fixed_points(num_fixed)
            all_perms.extend(perms)
        return all_perms
   
    def get_all_equilibrium_candidates(self):
        """
        Extract coefficients for all possible permutations.
        
        Returns:
            list of coefficient arrays
        """
        all_perms = self.possible_equilibria()
        candidates = []
        
        for pi in all_perms:
            coeffs = self.extract_coefficients(pi)
            candidates.append(coeffs)
        
        return candidates     

    def equilibrium_check(self, x):
        """
        Check equilibrium for given strategy profile x.
        
        Args:
            x: strategy profile (numpy array of length n)
        """
        print(f"Candidate: x = {x}")
        
        # Compute game matrix
        M = self.game_matrix(x)
        print(f"\nGame Matrix M:\n{M}")
        
        # Compute parity matrix
        P = self.parity(M)
        print(f"\nParity Matrix P:\n{P}")
        
        # Sum each row mod 2
        row_sums = np.sum(P, axis=1) % 2
        print(f"\nRow sums (mod 2): {row_sums}")
        
        # Calculate Inc(g,i) = [1 + x_i + row_sum[i]] mod 2
        inc = (1 + x + row_sums) % 2
        print(f"\nInc(x,i) = [1 + x_i + row_sum[i]] mod 2: {inc}")
        
        return M, P, row_sums, inc

    def check_all_equilibria(self):
        """
        Run equilibrium_check on all equilibrium candidates.
        Just display coefficients and inc values.
        """
        candidates = self.get_all_equilibrium_candidates()
        
        print(f"Total candidates: {len(candidates)}\n")
        
        for x in candidates:
            # Calculate inc
            M = self.game_matrix(x)
            P = self.parity(M)
            row_sums = np.sum(P, axis=1) % 2
            inc = (1 + x + row_sums) % 2
            
            # Count zeros and ones
            num_zeros = np.sum(inc == 0)
            num_ones = np.sum(inc == 1)
            
            print(f"  Coeffs: {x}, Inc: {inc} ({num_zeros} zeros, {num_ones} ones)")

    def partition_by_integers(self):
        """
        Partition equilibrium candidates by number of integer values in inc.
        """
        candidates = self.get_all_equilibrium_candidates()
        
        # Partition by number of integers in inc
        partitions = {i: [] for i in range(self.n + 1)}
        
        for x in candidates:
            # Calculate inc
            M = self.game_matrix(x)
            P = self.parity(M)
            row_sums = np.sum(P, axis=1) % 2
            inc = (1 + x + row_sums) % 2
            
            # Extract integer values from inc (convert to regular Python numbers)
            integers = [float(val) for val in inc if val == int(val)]
            num_integers = len(integers)
            
            partitions[num_integers].append((x, inc, integers))
        
        # Display results
        print(f"Total candidates: {len(candidates)}\n")
        print("="*70)
        
        for num_int in range(self.n, -1, -1):
            results = partitions[num_int]
            print(f"\n{num_int} integers in inc ({len(results)} candidates):")
            print("-"*70)
            for x, inc, integers in results:
                print(f"  Coeffs: {x}, Inc: {inc}")
                print(f"    Integers: {integers}")
        
        print("="*70)
        return partitions

    def get_integer_lists(self):
        """
        Extract and display coefficients with their increment integer values.
        Partition by number of integers in inc.
        """
        candidates = self.get_all_equilibrium_candidates()
        
        # Partition by number of integers in inc
        partitions = {i: [] for i in range(self.n + 1)}
        
        for x in candidates:
            # Calculate inc
            M = self.game_matrix(x)
            P = self.parity(M)
            row_sums = np.sum(P, axis=1) % 2
            inc = (1 + x + row_sums) % 2
            
            # Extract integer values from inc
            integers = [int(val) for val in inc if val == int(val)]
            num_integers = len(integers)
            
            partitions[num_integers].append((x, integers))
        
        # Display results
        print(f"Coefficients and Increment (Integer):\n")
        print("="*70)
        
        for num_int in range(self.n, -1, -1):
            results = partitions[num_int]
            print(f"\n{num_int} integers ({len(results)} candidates):")
            print("-"*70)
            for coeffs, integers in results:
                print(f"  Coeffs: {coeffs}, Increment (integer): {integers}")
        
        print("="*70)
        return partitions
        
    def equilibria(self):
        """
        List equilibrium candidates.
        If integers in inc contain 1, change the first 0 in coeffs to 1.
        """
        candidates = self.get_all_equilibrium_candidates()
        equilibria_list = []
        
        for x in candidates:
            # Calculate inc
            M = self.game_matrix(x)
            P = self.parity(M)
            row_sums = np.sum(P, axis=1) % 2
            inc = (1 + x + row_sums) % 2
            
            # Extract integer values from inc
            integers = [float(val) for val in inc if val == int(val)]
            
            # Make a copy of coefficients
            modified_coeffs = x.copy()
            
            # If integers contain 1, change first 0 in coeffs to 1
            if 1.0 in integers:
                for i in range(len(modified_coeffs)):
                    if modified_coeffs[i] == 0.0:
                        modified_coeffs[i] = 1.0
                        break
            
            equilibria_list.append(modified_coeffs)
        
        # Display results
        print(f"Total equilibria: {len(equilibria_list)}\n")
        
        for eq in equilibria_list:
            print(f"  {eq}")
        
        return equilibria_list

    def partition_equilibria(self):
        """
        Partition equilibria by number of mixed players.
        """
        print(game.A)
        equilibria_list = self.equilibria()
        
        # Partition by number of decimals
        partitions = {i: [] for i in range(self.n + 1)}
        
        for eq in equilibria_list:
            # Count decimal (non-integer) values
            num_decimals = sum(1 for val in eq if val != int(val))
            partitions[num_decimals].append(eq)
        
        # Display results
        print(f"\nPartitioning {len(equilibria_list)} equilibria by mixed players:\n")
        print("="*70)
        
        for num_dec in range(self.n, -1, -1):
            results = partitions[num_dec]
            print(f"\n{num_dec} mixed ({len(results)} equilibria):")
            print("-"*70)
            for eq in results:
                print(f"  {eq}")
        
        print("="*70)
        
        return partitions
    
    def equilibrium_count(self):
        """
        Count total equilibria using formula:
        - For n mixed (completely mixed): just count
        - For < n mixed: count × 2^(n - num_mixed - 1)
        """
        # Get partitions
        equilibria_list = self.equilibria()
        
        # Partition by number of decimals (mixed players)
        partitions = {i: [] for i in range(self.n + 1)}
        
        for eq in equilibria_list:
            num_decimals = sum(1 for val in eq if val != int(val))
            partitions[num_decimals].append(eq)
        
        # Calculate weighted count
        total_count = 0
        
        print(f"\nEquilibrium Count Formula:\n")
        print("="*70)
        
        # Handle completely mixed (n mixed) - no multiplication
        count_n = len(partitions[self.n])
        total_count += count_n
        print(f"{self.n} mixed (completely mixed): {count_n}")
        
        # Handle rest with 2^(n - num_mixed - 1)
        for num_mixed in range(self.n - 1, -1, -1):
            count = len(partitions[num_mixed])
            weight = 2 ** (self.n - num_mixed - 1)
            contribution = count * weight
            total_count += contribution
            
            print(f"{num_mixed} mixed: {count} × 2^({self.n}-{num_mixed}-1) = {count} × {weight} = {contribution}")
        
        print("="*70)
        print(f"Total equilibria count: {total_count}\n")
    
        return total_count
   
    def equilibrium_count_1(self):
        """
        Count total equilibria using formula, displaying all equilibria
        partitioned by number of mixed players with dividers.
        Uses formula: !n + sum_{l=1}^{n} C(n,l) * 2^{l-1} * !(n-l)
        """
        equilibria_list = self.equilibria()
        
        # Partition by number of mixed players
        partitions = {i: [] for i in range(self.n + 1)}
        for eq in equilibria_list:
            num_decimals = sum(1 for val in eq if val != int(val))
            partitions[num_decimals].append(eq)
        
        total_count = 0
        
        print(f"\nEquilibrium Count (n = {self.n}):\n")
        print("=" * 70)
        
        # Completely mixed: !n derangements
        num_mixed = self.n
        eqs = partitions[num_mixed]
        total_count += len(eqs)
        
        print(f"\n0 Pure, {self.n} Mixed:  !{self.n} = {len(eqs)}")
        print("-" * 70)
        if len(eqs) == 0:
            print("   (none)")
        else:
            for x in eqs:
                print(f"   {x}")
        print("-" * 70)
        
        # For l pure, (n-l) mixed: C(n,l) * 2^{l-1} * !(n-l)
        for l in range(1, self.n + 1):
            num_mixed = self.n - l
            eqs = partitions[num_mixed]
            
            from math import comb, factorial
            # Count derangements !(n-l)
            def subfactorial(k):
                if k == 0:
                    return 1
                return int(round(factorial(k) * sum((-1)**i / factorial(i) for i in range(k + 1))))
            
            derangements = subfactorial(num_mixed)
            binom = comb(self.n, l)
            pure_combos = 2 ** (l - 1)
            contribution = binom * pure_combos * derangements
            total_count += contribution
            
            print(f"\n{l} Pure, {num_mixed} Mixed:  C({self.n},{l}) × 2^({l}-1) × !{num_mixed} = {binom} × {pure_combos} × {derangements} = {contribution}")
            print("-" * 70)
            if len(eqs) == 0:
                print("   (none)")
            else:
                for x in eqs:
                    print(f"   {x}")
            print("-" * 70)
        
        print("\n" + "=" * 70)
        print(f"Total: !{self.n} + Σ C({self.n},l)·2^(l-1)·!({self.n}-l) = {total_count}\n")
        
        return total_count

    def perm_parity(self, pi):
        """
        Returns parity of permutation pi (1-indexed).
        0 = even, 1 = odd.
        """
        perm = [p - 1 for p in pi]
        n = len(perm)
        visited = [False] * n
        swaps = 0
        for i in range(n):
            if not visited[i]:
                j, cycle_len = i, 0
                while not visited[j]:
                    visited[j] = True
                    j = perm[j]
                    cycle_len += 1
                swaps += cycle_len - 1
        return swaps % 2

    def display_equilibria_check(self):
        """
        Display equilibria with their game matrices, parity matrices,
        permutation and upsteps/downsteps.
        Excludes n mixed and n-1 mixed equilibria.
        Shows coefficient matrix A at the top.
        """
        equilibria_list = self.equilibria()
        
        print(f"\nEquilibria with Parity (excluding {self.n} and {self.n-1} mixed):\n")
        print("=" * 70)
        
        print("Delta Sort Order Coefficient Matrix A:")
        for row in self.A:
            print(f"  {row}")
        print("=" * 70)
        
        count = 0
        for x in equilibria_list:
            num_mixed = sum(1 for val in x if val != int(val))
            
            if num_mixed == self.n or num_mixed == self.n - 1:
                continue
            
            # Recover permutation
            pi = []
            for j in range(self.n):
                if x[j] == int(x[j]):
                    pi.append(j + 1)
                else:
                    for i in range(self.n):
                        if i != j and np.isclose(self.A[i, j], x[j]):
                            pi.append(i + 1)
                            break
            
            up, down = self.count_upsteps(pi)
            
            M = self.game_matrix(x)
            P = self.parity(M)
            
            count += 1
            print(f"\n{count}. π = {pi}  |  Up: {up}  Down: {down}")
            print(f"   Coeffs: {x}")
            
            print(f"   Game Matrix M:")
            for row in M:
                print(f"     {row}")
            
            print(f"   Parity:")
            for row in P:
                print(f"     {row}")
        
        print("\n" + "=" * 70)
        print(f"Total displayed: {count} equilibria\n")
  
    def display_equilibria_all(self):
        """
        Compact display: permutation, permutation parity, and coeffs.
        No game matrix or parity matrix.
        Excludes n mixed and n-1 mixed equilibria.
        Shows coefficient matrix A at the top.
        """
        equilibria_list = self.equilibria()
        
        print(f"\nCompact Equilibria Display (excluding {self.n} and {self.n-1} mixed):\n")
        print("=" * 70)
        print("Delta Sort Order Coefficient Matrix A:")
        for row in self.A:
            print(f"  {row}")
        print("=" * 70)
        
        count = 0
        for x in equilibria_list:
            num_mixed = sum(1 for val in x if val != int(val))
        
            if num_mixed == self.n:
                continue
            # Recover permutation from coefficients
            pi = []
            for j in range(self.n):
                if x[j] == int(x[j]):  # fixed point
                    pi.append(j + 1)
                else:
                    # Find which player's coefficient this is in column j
                    for i in range(self.n):
                        if i != j and np.isclose(self.A[i, j], x[j]):
                            pi.append(i + 1)
                            break
            
            par = self.perm_parity(pi)
            
            count += 1
            print(f"\n{count}. π = {pi}  |  Perm Parity: {'odd' if par else 'even'} ({par})")
            print(f"   Coeffs: {x}")
        
        print("\n" + "=" * 70)
        print(f"Total displayed: {count} equilibria\n")

    def count_upsteps(self, pi):
        """Returns (upsteps, downsteps) for a 1-indexed permutation."""
        up = sum(1 for i in range(len(pi)) if pi[i] > i + 1)
        down = sum(1 for i in range(len(pi)) if pi[i] < i + 1)
        return up, down
    
    def display_equilibria_compact(self):
        """
        Compact display: permutation, upsteps/downsteps, and coeffs.
        No game matrix or parity matrix.
        Excludes n mixed and n-1 mixed equilibria.
        Shows coefficient matrix A at the top.
        """
        equilibria_list = self.equilibria()
        
        print(f"\nCompact Equilibria Display (excluding {self.n} and {self.n-1} mixed):\n")
        print("=" * 70)
        print("Delta Sort Order Coefficient Matrix A:")
        for row in self.A:
            print(f"  {row}")
        print("=" * 70)
        
        count = 0
        for x in equilibria_list:
            num_mixed = sum(1 for val in x if val != int(val))
            
            if num_mixed == self.n:
                continue
            
            # Recover permutation from coefficients
            pi = []
            for j in range(self.n):
                if x[j] == int(x[j]):  # fixed point
                    pi.append(j + 1)
                else:
                    for i in range(self.n):
                        if i != j and np.isclose(self.A[i, j], x[j]):
                            pi.append(i + 1)
                            break
            
            up, down = self.count_upsteps(pi)
            
            count += 1
            print(f"\n{count}. π = {pi}  |  Upsteps: {up}  Downsteps: {down}")
            print(f"   Coeffs: {x}")
        
        print("\n" + "=" * 70)
        print(f"Total displayed: {count} equilibria\n")


#------------------------------------Usage starts here


n = 5
game = ProductTwoActionGame(n) 
#The method display_equilibira_all shows the equialibira of n-1,n-2,...0 mixing with the permutation and parity of permutation
# The method display_equilibira_compact only the  equilibira of cases n-2,n-3,...,0 mixing players. 
# this method also shows the permutation and the parity of the permutation. 
# To see that the bound is achieved the method equilibirum_count properly counts each equilibirum 
# To check that each equilibrium is correct we can run instantiate the method display_equilibira_check
print(game.display_equilibria_check())
#print(game.display_equilibria_all())
print(game.display_equilibria_compact())
print(game.equilibrium_count_1())

