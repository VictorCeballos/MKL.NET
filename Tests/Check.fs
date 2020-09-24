﻿module Check

open System
open System.Diagnostics

let info fmt =
    let sb = System.Text.StringBuilder()
    Printf.kbprintf (fun () -> sb.ToString() |> Information) sb fmt

let label s = Label s

let message format =
    Printf.kprintf (fun msg ->
        function
        | Failure f -> Failure(Message msg + ": " + f)
        | Faster (_,s) -> Faster(msg,s)
        | r -> r
    ) format

let isTrue (actual:bool) =
    if actual then Success
    else Failure(Normal "actual is false.")

let isFalse (actual:bool) =
    if actual then Failure(Normal "actual is true.")
    else Success

let equal (expected:'a) (actual:'a) =
    let rec equal (expected:obj) (actual:obj) =
        let equalArray (e:'b[]) (a:'b[]) =
            if e.Length = a.Length then
                Array.fold2 (fun (i,s) e a ->
                    match s with
                    | Success ->
                        i+1, equal e a
                    | f -> i,f
                ) (-1,Success) e a
                |> function | i, Failure t -> Failure(Normal "Index: " + Numeric i + ". " + t)
                            | _, r -> r
            else
                Failure(Normal "Length differs. expected: " + Numeric e + " actual: " + Numeric a)
        let inline equalDefault (e:'b) (a:'b) =
            let e = (string e).Replace("\n","")
            let a = (string a).Replace("\n","")
            Failure(Normal "actual is not equal to expected.\n     actual: " + Numeric a + "\n   expected: " + Numeric e)
        match expected, actual with
        | (:? float as e), (:? float as a) ->
            if e=a || Double.IsNaN a && Double.IsNaN e then Success
            else equalDefault e a
        | (:? float32 as e), (:? float32 as a) ->
            if e=a || Single.IsNaN a && Single.IsNaN e then Success
            else equalDefault e a
        | (:? (float[]) as e), (:? (float[]) as a) -> equalArray e a
        | (:? (float32[]) as e), (:? (float32[]) as a) -> equalArray e a
        | e, a ->
            if a=e then Success
            else equalDefault e a
    equal expected actual

let between (actual:'a) (startInclusive:'a) (endInclusive:'a) =
    if actual < startInclusive then
        Failure(Normal "actual (" + Numeric actual + ") is less than start (" + Numeric startInclusive + ").")
    elif actual > endInclusive then
        Failure(Normal "actual (" + Numeric actual + ") is greater than end (" + Numeric endInclusive + ").")
    else Success

let greaterThan (expected:'a) (actual:'a) =
    if actual > expected then Success
    else Failure(Normal "actual (" + Numeric actual + ") is not greater than " + Numeric expected + ".")

let close accuracy (expected:'a) (actual:'a) =
    let rec close (expected:obj) (actual:obj) =
        let closeArray (e:'b[]) (a:'b[]) =
            if e.Length = a.Length then
                Array.fold2 (fun (i,s) e a ->
                    match s with
                    | Success ->
                        i+1, close e a
                    | f -> i,f
                ) (-1,Success) e a
                |> function | i, Failure t -> Failure(Normal "Index: " + Numeric i + ". " + t)
                            | _, r -> r
            else
                Failure(Normal "Length differs. expected: " + Numeric e + " actual: " + Numeric a)
        let inline closeDefault (e:'b) (a:'b) =
            Failure(Normal "Expected difference to be less than "
            + Numeric(Accuracy.areCloseRhs accuracy a e)
            + Normal ", but was " + Numeric(Accuracy.areCloseLhs a e)
            + Normal ". actual=" + Numeric a + Normal " expected=" + Numeric e
            )
        match expected, actual with
        | (:? float as e), (:? float as a) ->
            if Accuracy.areClose accuracy e a
             || Double.IsNaN a && Double.IsNaN e
             || Double.IsPositiveInfinity a && Double.IsPositiveInfinity e
             || Double.IsNegativeInfinity a && Double.IsNegativeInfinity e
             then Success
            else closeDefault e a
        | (:? float32 as e), (:? float32 as a) ->
            if Accuracy.areClose accuracy e a
             || Single.IsNaN a && Single.IsNaN e
             || Single.IsPositiveInfinity a && Single.IsPositiveInfinity e
             || Single.IsNegativeInfinity a && Single.IsNegativeInfinity e
             then Success
            else closeDefault e a
        | (:? (float[]) as e), (:? (float[]) as a) -> closeArray e a
        | (:? (float32[]) as e), (:? (float32[]) as a) -> closeArray e a
        | _ -> failwithf "Unknown type %s" (actual.GetType().Name)
    close expected actual

let faster (expected:unit->'a) (actual:unit->'a) =
    let t1 = Stopwatch.GetTimestamp()
    let aa,ta,ae,te =
        if t1 &&& 1L = 1L then
            let aa = actual()
            let t1 = Stopwatch.GetTimestamp() - t1
            let t2 = Stopwatch.GetTimestamp()
            let ae = expected()
            let t2 = Stopwatch.GetTimestamp() - t2
            aa,t1,ae,t2
        else
            let ae = expected()
            let t1 = Stopwatch.GetTimestamp() - t1
            let t2 = Stopwatch.GetTimestamp()
            let aa = actual()
            let t2 = Stopwatch.GetTimestamp() - t2
            aa,t2,ae,t1
    match equal aa ae with
    | Success -> Faster("",if te>ta then float(te-ta)/float te else float(te-ta)/float ta)
    | fail -> fail

/// Chi-squared test to 6 standard deviations.
let chiSquared (expected:int[]) (actual:int[]) =
    if actual.Length <> expected.Length then
        Failure(Normal "actual and expected need to be the same length.")
    elif Array.exists (fun i -> i<=5) expected then
        Failure(Normal "expected frequency for all buckets needs to be above 5.")
    else
        let chi = Array.fold2 (fun s a e ->
            let d = float(a-e)
            s+d*d/float e) 0.0 actual expected
        let mean = float(expected.Length - 1)
        let sdev = sqrt(2.0 * mean)
        let SDs = (chi - mean) / sdev
        if abs SDs > 6.0 then
            Failure(Normal "chi-squared standard deviation = " + Numeric(SDs.ToString("0.0")))
        else Success
