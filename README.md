## TOC 
- [Vector](#Vector)
- [Iterator](#Iterator)
- [String](#String)
- [Tree](#BTreeMap)
- [HashMap](#Hashmap)
- [Result & Option](#Result-&-Option)
- [Match](#Match)
- [Important Traits](#Important-Traits)
- [Important Crates](#Important-Crates)
- [File Structure](#File-Structure)
- [Shorthands](#Shorthands)
- [Generics](#Generics)
- [Lifetimes](#Lifetimes)
- [Closures](#Closures)
- [References](#References)
- [Threads](#Threads)
- [Channels](#Channels)
- [Builder Pattern](#Builder-Pattern)
- [Macros](#Macros)
- [Futures](#Futures)
- [Databases](#Databases)
- [Web Services](#Web-Servers)
- [CLI](#CLI)

## Vector

```rust
vec![0,1,2] or Vec::new()

v.push()                    // LIFO stack
v.pop() -> Option           // LIFO stack
v.len() -> usize
v.insert(*i*, *elem*)
v.remove(*i*) -> elem

v.append(vec2)
v.truncate(*l*)
v.sort()
v.dedup()                   // unique set
v.retain(|&x| x % 2 == 0);  //filter
v.splice(..2, v2.iter().cloned()) -> &removed times )
v.split_at_mut(i);          // splits vec returning mut vecs
v.contains(&30) -> bool
```

## Iterator

```rust
v.iter()

iter.next()
iter.count()
iter.last()
iter.nth()

iter.map()
iter.filter()
iter.find() -> Option
iter.any() -> bool
iter.fold() 
iter.flatten() 

```

## String

```rust
// When you need Owned string data (e.g. passing btw threads)                                                             
// &String == &str, since String impl "Deref str" 
String::from("hi")
format!("{0} hi", x)

// String -> &str
s.as_str(), as_mut_str()

s.push(char)
s.truncate()
s.pop()...
```

## String Slice 

```rust
// When you only need a view of a string 

// &str -> String
s.to_string()
*s*.chars() -> Iterable
s.parse() -> Result
s.len()
s.is_empty()
```

- Trait `AsRef<str>` : if `s` is already a string, then it returns ptr to itself, but if it's already a pointer to str, then it doesn't do anything (no op). 
- Has method `.as_ref()` *Saves you from implementing fns for both strings and slices*

## BTreeMap
```rust
use std::collections::BTreeMap;
BTreeMap::new()

m.get(key)
m.insert(key,val)
m.remove(key)
m.contains_key(key)
m.keys()
m.values()
m.len()
m.is_empty()
```

## Hashmap
```rust
use std::collections::HashMap;
HashMap::new()

m.insert(k,v)
m.contains_key(k)
m.remove(k) -> Option
m.get(k) -> Option

// Iterate over everything.
for (k, v) in book_reviews {}
for v in m.values_mut()

// Changing values
*m.get_mut(&k).unwrap()...

// Insert key if not exist
let v = m.entry(k).or_insert(v_default); // -> mut V
*v+= 1;
```
## Result & Option

```rust
pub enum Result<T,E> {
    Ok(T),
    Err(E),
}

pub enum Option<T> {
    Some(T),
    None,
}

// Result -> Option
get_result.ok()?;
// Option -> Result
ok_or("An Error message if the Option is None")?; // ok_or converts None into Error

// Chaining fn if Ok, else return Err
.and_then(fn)
```

## Match

```rust
// Match Guards

let message = match maybe_digit {
    Some(x) if x < 10 => process_digit(x),
    Some(x) => process_other(x),
    None => panic!(),
};

let i : Cell<i32> = Cell::new(0);
match 1 {
    1 | _ if { i.set(i.get() + 1); false } => {}
    _ => {}
}
assert_eq!(i.get(), 2);
```

## Important Traits

- `from` and `into`: *Two sides of the same coin, returns the same Type the trait is impl on. `Into` is free after impl `From`*

    let my_string = String::from(my_str);
    let num: Number = int.into();

- `PartialOrd`: Allows the > , < , = operators in T
- `std::ops::AddAssign`: Allows arithmetic operations in T
- `Copy`: Allows copy for primitives, rather than just move
- `Default`: `.. Default::default()` fills in defaults for a struct/obj;

## Basic Crates
- `std::fs::File`, `std::io::Read/Write`, `std::time`
- `Failure`: makes custom error creation easier
- `Itertools`: interleave(), interlace() *for strings* , combines iterators together in woven way
- `Rayon`: Thread, pool handling by allowing **parallel iterators.** `intoParallelIterator`
- `Tokio`: Provides multithread Future executor for Streams; handles async well, `tokio::run(a_future)`

## File Structure

Cargo.toml cheatsheet

```toml
// Generic Cargo.toml

// library 
[lib] // some library
name = "..."
path = "..."

// binaries 
// access libaries with `extern crate lib_name;`
[[bin]]
name = "..."
path = "..."

[dependencies]
dep_name = "version_num"

// Alternative to saying dep_name = {version =..., features = ["..."]}
[dependencies.dep_name]
version = "..."
features = ["..."]  // features are conditional compilation flags
```

## Shorthands

### Ellipsis

- `..`: inclusive range operator,  e.g. `start..=end`
- `..`: struct update syntax, i.e. remaining fields not explicitly set should have the same value as the fields in the given instance. `..Default::default()` or `..user1`

## ? Shorthand

*? applies to a `Result` value, and if it was an `Ok`, it **unwraps** and gives the inner value (Directly uses the into() method), or it returns the Err type from the current* function.* 

```rust
// Unwraps a result, or propagates an error (returns the `into` of an Error)
// ? needs the From trait implementation for the error type you are converting
Ok(serde_json::from_str(&std::fs::read_to_string(fname)?)?)
```

### Use custom errors for new structs

*To handle proper error handling*
```rust
use serde_derive::*;

#[derive(Debug)]
pub enum TransactionError {
    LoadError(std::io::Error),
    ParseError(serde_json::Error),
}

impl From<std::io::Error> for TransactionError {
    // we have to impl this required method
                                // Self: the type we are working with right now, in this case its TransactionError
    fn from(e: std::io::Error) -> Self {
        TransactionError::LoadError(e)
    }
}

#[derive(Deserialize, Serialize, Debug)]
pub struct Transaction { ... }
```

## Error handling

1. Put all error traits & implementations in `error.rs`; Put all functions in `lib.rs`
2. Create a lib in cargo.toml, point [main.rs](http://main.rs) to library crate  `extern crate lib_name`

```toml
// Allows us to have both a library and an application
[lib]     # library name, path, etc
[[bin]]   # binary target name, path etc. Can have multiple binaries!
```

3. Inside `[lib.rs](http://lib.rs)` import error.rs with `mod error`  `use error::...`

**Failure Trait:** *Better failure handling, without having to create my own error type*

[failure](https://rust-lang-nursery.github.io/failure/)

- When using it to write a library: create a new Error type
- When using it to write an application: have all functions that can fail return failure::Error

## Generics & Traits

*Generic Types*
```rust
// Declaration
pub const fn new() -> Vec<T> 

// Usage
path::<datatype>::method()
let a = Vec::<bool>::new()
```

*Generic Functions* 
```rust
// Declaration
fn collect<B>(self) -> B

// Usage
method::<datatype>()
let a = iter.collect::<Vec<i32>>();
```


*Generic Type, Functions & Trait imps*
```rust
// TRAITS
pub struct Thing<T> { foo: T, ...}...
impl <T> Thing<T> {...}
impl <T> TraitA for Thing<T> {...}

// Limit generics to certain Traits & provide addn traits it needs
impl <T:TraitA + TraitB> ...
// Or
impl <T> ... where T:SomeTrait {...}

// Trait requirements: if type impl B, it might impl A
trait B: A

// 1. Refactor the multi-traits by creating a new Trait
pub trait CombiTrait: TraitA + TraitB {..// any fns ..}
// 2. Impl CombiTrait for all T that has traitA/B, otherwise get error
impl <T:TraitA + TraitB> CombiTrait for T {}

// FUNCTIONS
// It makes the type available for the function
fn print_c<I:TraitA>(foo:I) {
```

## Lifetimes

*Think of lifetimes as a constraint, bounding output to the same lifetime as its input(s)*

- By default, lifetimes exist only within the scope of a function
- Only things relating to references need a lifetime, e.g. struct containing a ref, or strings
- `'static` ensures a lifetime for the entirety of the program

```rust
//FUNCTION output lifetime binds to input variables: 
// 1. We assign the lifetime 'a' to variables s, t 
// 2. We constrain the output lifetime to 'a' as well
// 3. Meaning the output now has to live as long as the smaller of s or t
fn foo<'a>(s: &'a str; t: &'a &str) -> &'a str {...}

// STRUCTs containing refs, require lifetimes, 
// To ensure any reference to Foo doesn't outlive reference to x (internal value)
struct Foo<'a> {	x: &'a i32, }
impl<'a> Foo<'a> { ... }

// ELISIONS: i.e. inferences & allowed shorthands
fn print(s: &str); // elided
fn print<'a>(s: &'a str); // expanded

// If ONE input lifetime, elided or not, that lifetime is assigned to all elided lifetimes in the return values of that function.
fn substr(s: &str, until: u32) -> &str; // elided
fn substr<'a>(s: &'a str, until: u32) -> &'a str; // expanded
// fn frob(s: &str, t: &str) -> &str; // ILLEGAL, two inputs
  ```

## Closures

- `FnOnce`: consumes variables it captures. Moves vars into the closure when its defined. FnOnce can only be called once on the same variables
- `FnMut`: can change env bc it mutably borrows values. *recommended for working with Vecs*
- `Fn`: borrow values from env immutably
```rust
// FnOnce:
pub fn foo<F>(&mut self, f: F)
    where F: FnOnce(&mut String), {...}

// FnMut: The function itself is going to change over time
pub fn edit_all<F>(&mut self, mut f: F)
    where F: FnMut(&mut String), {...}

// Boxed closure type, usually used in threads, 
// allowing different fn to be sent to the same channel
Box<Fn(&mut String) + TraitB >
```

## References
```rust
&i32        // a reference
&'a i32     // a reference with an explicit lifetime
&'a mut i32 // a mutable reference with an explicit lifetime

Box<T>      // is for single ownership.
Rc<T>       // is for multiple ownership.
Arc<T>      // is for multiple ownership, but threadsafe.
Cell<T>     // is for “interior mutability” for Copy types; that is, when you need to mutate something behind a &T.
```

### Reference count

*Single thread use only: doesn't impl sync and send*

- `Rc<>`: generic struct, like a garbage collector.  `(reference count, ptr to data)` When you clone an Rc, it just increments the count without copying the data; when it reaches 0 it drops its internal pointer.
- `Rc::new(s)`: Turns into reference counted obj, & acts as a reference, usable with `.clone()`. S has to be immutable.

```rust
// Make sure to specific the Rc type when a var is going to be reference counted
s: Rc<String> // a generic type, acts as a reference to String
```

### RefCell: a mutable borrow

*Single thread use only: doesn't impl sync and send*

- Internally: guards the borrow checker, makes sure changes are correct.
- Externally: pretends that it doesn't change. To the borrow checker, it mimics immutable obj.
- `Rc<<RefCell<s>>` , `Rc::new(RefCell::new(s))`, `let ... = s.borrow_mut()` and lastly `drop(s)` to disable the guard.

## Threads

- Be careful: The main fn can terminate before new threads finish!

Passing shared data foo btw threads: 
```rust
let foo = Arc::new<Mutex<data>>;
let (foo_1, foo_2) = (foo_1.clone(), foo_2.clone());

// the **Move** closure **owns** its values: not subject to borrow checker, so variables inside can live outside its og value
let thread_handle_1 = thread::spawn( move || fn_1(foo_1) )
let thread_handle_2 = thread::spawn( move || fn_2(foo_2) )

let result_1 = thread_handle_1.join.unwrap();

// ========================
// To use the Mutex var in fn_1:
{ 
    // retrieve lock inside a scope so the lock is immediately dropped
    foo_1.lock().unwrap();
}
```

```rust
// Diff way to wait for threads to complete
let mut children = Vec::new();
let child = thread::spawn(...);
for child in children { child.join().expect("thread panic");
```

- When you need to move **primitive** variables to be accessible inside threads, thanks to `Copy`:
`spawn( move || { println!("{}", n); });`

### Arc, Mutex

If `Copy` is not implemented for the var (e.g. moving a String btw threads), use: 

- `arc`: atomic reference count.
- `mutex`: a guard on mutation, allows mutation when ppl have a `lock` on it
- `lock()`: acquires the mutex's lock, ensuring its only held by 1 obj, returns Result<>.

```rust
use std::sync::{Arc, Mutex};
fn foo () {
    let m = Arc::new(Mutex::new(String::from("xyz")));

    // Create a new arc mutex by copying original arc mutex
    // but really clone just increments the rc count
    let m2 = m.clone();
    // use m2 in new thread
    spawn ( ... let mut s = m2.lock().unwrap(); ... )

    // use m in original thread
    let s = m.lock().unwrap();
}
```

When to use Mutex in Arc?
```rust
// In a multi-thread game engine, you might want to make an Arc<Player>. 
// The Mutexes protect the inner fields that are able to change (items). 
// while still allowing multiple threads to access id concurrently!

struct Player {
    id: String,
    items: Mutex<Items>,
}
```

### MPSC Channels

- A function returning: `Sender<T>` and `Receiver<T>` endpoints, where T is type of msg to be transferred, allowing async communications between threads
- `mpsc`: means Multi Producer Single Consumer. We can clone sender, but not receiver.
- Drop(channel senders) will close the channel automatically
- **Fns have to be Boxed** before sending across channel

*MPSC uses unbounded channels: so if read faster than write, then we can run out of memory. Medium performance.* 

```rust
// Assume ThreadA --[channel1]--> ThreadB --[channel2]--> ThreadC

// Naming convention: 
    // B_tx: Sender<Vec<u8>>, transmitting to thread b, from a 
    // ThreadB_rx: Receiver<Vec<u8>>, receiving in thread b
let (b_tx, b_rx) = mpsc::channel();
let (c_tx, c_rx) = mpsc::channel();

// pass in the sender to thread 1
let thread_a = thread::spawn(move || fn_1(b_tx));
let thread_b = thread::spawn(move || fn_2(b_rx, c_tx));
let thread_c = thread::spawn(move || fn_3(c_rx));

// ========================
// To use channels in fn_2:
b_rx.recv().unwrap(); // a blocking call
c_tx.send(Vec::from(*some_byte_slice*));

// Sending could fail, so best to handle if producer errors:
if b_tx.send(..).is_err() { break; }
```

Another good pattern:
```rust
//Good pattern
std::sync::mpsc::channel;

fn foo() {
    let (ch_s, ch_r) = channel<...>();

    // channel that sends a done signal, instead of waiting
    let (done_s, done_r) = channel<()>();

    std::thread::spawn (move || 
        loop {
            match ch_r.recv() {
                Ok(_) => {...}
                Err(_) => { done_s.send(()).unwrap() }
        }
    }

  // sending stuff through channels
    ch_s.send(...)...
  let ch_2 = ch_s.clone();
    ch_2.send(...)...

    // Need to manually drop the channels or infinite loop occurs
    drop(ch_s);
  drop(ch_2);

    done_r.recv().ok();
}
```

#### ThreadPools
```rust
pub struct ThreadPool {
    ch_s: Option<mpsc::Sender<Box<Fn() + Send>>>,
    n: u32,
    ch_done: mpsc::Receiver<()>,
}

impl ThreadPool {
    pub fn new(n: u32) -> Self {
        let (ch_s, ch_r) = mpsc::channel();
        let a = Arc::new(Mutex::new(ch_r)); // multi receiver
        let (ch_done_s, ch_done) = mpsc::channel();

        for _ in 0..n {
            let a2 = a.clone();
            let ch_done_2 = ch_done_s.clone();
            // This thread will loop: waiting on the receiver
            // for the next job that it is going to do
            std::thread::spawn(move || loop {
                let m = a2.lock().unwrap();
                let f: Box<Fn() + Send> = match m.recv() {
                    Ok(f) => f, 
                    Err(_) => {
                        ch_done_2.send(()).ok();
                        return;
                    }
                };
                // drop our hold on the mutex before we run f
                // otherwise only one fn can run at a time
                drop(m);
                f();
            });            
        }

        ThreadPool{ch_s:Some(ch_s) , n, ch_done}
    }

    pub fn run<F:Fn() + Send + 'static>(&self, f:F) {
        if let Some(ref ch_s) = self.ch_s {
            ch_s.send(Box::new(f)).unwrap();
        }
    }

    // consumes self at the end
    pub fn wait(mut self) {
        self.ch_s.take();  // drops our sender
        for _ in 0..self.n {
            self.ch_done.recv().unwrap(); // waits for n done messages to come back
        }
    }
}

fn main() {
    let tp = ThreadPool::new(n_of_threads);
    for something_i_want_n_times {
        tp.run(...)
    }
    tp.wait();
}
```

*Alternative is crossbeam channels: which is bounded and unbounded.*

### Crossbeam Channels
```rust
// Assume ThreadA -> [channel1] -> ThreadB   // only some metadata this time
//        ThreadA -> [channel2] -> ThreadC

// Main Usage: you typically want to bound the writing threads since read is always faster
use crossbeam::channel::{bounded, unbounded}
let (b_tx, b_rx) = unbounded;
let (c_tx, c_rx) = bounded(1024); // blocking after 1024 message cap

let thread_a = thread::spawn(move || fn_1(b_tx, c_tx)); 
let thread_b = thread::spawn(move || fn_2(b_rx)); 
let thread_c = thread::spawn(move || fn_3(c_rx)); // the slower write thread

// Sender/Receiver params work the same as MPSC
```

## Builder Pattern

```rust
pub enum Property {
    Simple(&'static str, String), // e.g. x=5, y = 11
    Style(&'static str, String),  // like css styles, e.g. border: ...
    Transform(String),           
}

pub struct SvgTag {
    pub kind: &'static str,
    pub properties: Vec<Property>,
}

impl SvgTag {
    pub fn new(kind: &'static str) -> Self {
        SvgTag {
            kind,
            properties: Vec::new(),
        }
    }

    // Display: formats values, so we can use to_string() method on v
    // Takes mutable self, takes ownership of self, changes it, then returns self
    // fn that calls this fn loses access on it
    pub fn property<V: Display>(mut self, k: &'static str, v:V) -> Self {
        self.properties.push(Property::Simple(k, v.to_string()));
        self
    }

    pub fn style<V: Display>(mut self, k: &'static str, v:V) -> Self {
        self.properties.push(Property::Style(k, v.to_string()));
        self
    }

    // sets the x property
    pub fn x<V:Display>(self, v: V) -> Self {
        self.property("x", v)
    }

    pub fn y<V:Display>(self, v: V) -> Self {
        self.property("y", v)
    }
}

// So we can simply do: 
let a = SvgTag::new("svg").w("60px").h("80px");
```

### Defaults
```rust
struct Foo {...}

impl Default for Foo {
    fn default() -> Self {
        Foo { ..:.., }
    }
}
...
let f = Foo { ..Default::default()}
```

### Wrappers Traits

If I want to compose structs that contain certain (different?) traits. Create a wrapper struct. *This is usually done with iterators etc.*
```rust
pub struct TraitWrap<A:TraitA, B:TraitB> {
    a: A,
    b: B,
}

impl <A: TraitA, B:TraitB> TraitA for TraitWrap<A,B> {
        fn ... 
}
```

## Macros

### Common Metavariables

- `ident`: identifier, e.g. `x`, `foo`
- `item`: item, e.g. `fn foo() {}`, `struct Bar`
- `expr`: expression, e.g. `2+2`, `if x else y`, `f(42)`
    - May only be followed by one of: `=>`  `,` `;`
    - Expressions evaluate TO a value
- `stmt`: single statement, e.g. `let x = 3`
    - May only be followed by one of: `=>`  `,` `;`
    - Statements return a value
- `block`: block of code (stmts), e.g. `{let .. ; let ... return}`
- `pat`: pattern, e.g. `Some(t)`, `(17, 'a')`, `_`
- `path`: qualified name, e.g. `T::SpecialA`
- `ty`: type, e.g. `i32`, `Vec<>`, `&T`
- `tt`: single token tree, e.g. `()`, `[]`, or `{}`
- `meta`: meta item in attrs, e.g. `cfg(attr = "...")`

### Declarative Macros

Macros that work like match statements on your code snippets. It matches on illegal code, to replace them with legal Rust code.

```rust
// This macro should be made avail whenever the crate 
// where the macro is defined, is brought into scope
#[macro_export]

//macro decl
                        // name of macro 
macro_rules! vec {
        // similar to a match expression
        // if pattern matches, the following code executes
        
        //1. a () always encompasses the whole pattern
        //2. $(...) contains values that match the pattern within the parens for use within the replacement code
        //3. expr, a metavariable, will match any Rust expression
        //4. $x is now the name of the matched expression
        //5. , indicates that "," could appear after the code that matches the expression.
        //6. * means pattern can repeat, i.e. standard regular expression
        // So Vec![1,2,3] means 1 and 2 and 3 are respectively the $x
    ( $( $x:expr ),* ) => {
        {
            let mut temp_vec = Vec::new();
                        // $()* is generated for each part that matches $()
            $(
                temp_vec.push($x);
            )*
            temp_vec
        }
    };
}
```

### Macro refactoring

Calling a macro within a macro is a good way to refactor the macro code
```rust
// SvgTag::new("svg").child(SvgTag::new("rect").x(7).y(7).w(9).h(9));
// svg! { svg => {rect x=7, y=7, w=9, h=9} };

// Step 1: builds the children
//  { svg => {rect x=7, y=7, w=9, h=9} }
//     $s   $c
    ($s:ident => $($c:tt)*) => {
                //p matches with "rect"
        SvgTag::new(stringify!($s))
            $(.child(svg!$c))*   //1. adds the child, recursively calls itself to be able to add children
            //outer macros make all the changes, and then any inner macros in the result will then be evaluated
    };

// Step 2/c: builds the properties of a single child
// rect x=7, y=7, w=9, h=9
    ($s:ident $($p:ident = $v:expr),* ) => {
        SvgTag::new(stringify!($s))
            $(.$p($v))*
    };
```

### Procedural Macros

Functions that operate on input code, outputs new code
```rust
# This package does procedural macros
# and can't be used for anything other than proc macros
# this runs before other packages at compile time
[lib]
proc_macro = true

// In src/lib.rs
use proc_macro;

#[some_attribute]
pub fn some_name(input: TokenStream) -> TokenStream {
}
```

### Derive Macros

**Good convention**: for structs, to create setter helper fn, e.g. `set_field1` `set_field2` .

```rust
// using it
#[derive(MyMacro)]

// defining it
#[proc_macro_derive(MyMacro)]
pub fn mymacro_derive(input: TokenStream) -> TokenStream {
        let mut top = input.into_iter();
    let mut ttype: TokenTree = top.next().unwrap();
        format!(...).parse().unwrap()
                            // parse ret. Result, which is unrwapped
}

// helper setter fns for struct fields
// 
fn parse_row<I: Iterator<Item=TokenTree>>(i: &mut I) -> Option<Row> {...}

pub fn mymacro_derive ... {
    let mut row_iter = if let Some(TokenTree::Group(g)) = top.next() {
        g.stream().into_iter()
    } else {
        panic!();
    };

    // next step: create `rows` Vec
    // next step: parse_row  the rows into `rows`

    // Create a setter function per struct field
    let lines:String = rows.iter().map ( |r| {
        format!("fn set_{0}(&mut self, v:{1} {{
            self.{0} = v;
                }} \n", r.name, r.ttype)
    }).collect();

    // Create trait wrapper: "impl name { *lines of code* }"
    let res = format!(
        "impl {} {{ 
        {}
   }}", name, lines);
    
   eprintln!("Res == {}", res);

   res.parse().unwrap()
}
```
- `cargo expand`: expands macro code to be more readable. Install through `cargo install cargo-expand`

## Futures

A Future: a mechanism (action) for having an obj you know will return a delayed single result, in the future.

Poll: enum with variants `Pending` and `Ready<T>`

Runtime/Exector: Rust doesn't complete promises, you need something to poll the future, i.e. an executor like block_on

Think of futures are cheap threads, when you don't have as many threads. How it works is different though. Threads are run on separate processes. Futures are: 

- Obj with a function that will return if there is something it can't do, e.g. waiting on tcp , or file load
- Hidden in a struct, which some other operator has to call to ask if it needs to do something
- Like a Javascript promise: Returns something that you can call methods on; Easier to keep things in an asynchronous.
- Is a Rust Library, not embedded in the language itself.
- Zero cost abstraction: if you don't use it, it doesn't cost me to include it. (Conversely, green threads in Go do cost)
- Async / Await (unstable) is here already (Jan 2020)

### Simple Future
```rust
    pub struct SimpleFuture {
        n: i32, // or whatever other return type upon completion
    }
    
    impl Future for SimpleFuture {
        type Output = i32;
        
        // Req function: attempt to resolve future to final value, 
        // registering the current task for wakeup if value is not yet available
                    // Pin dereferences to Self, guarantees we can still access ptrs to internal ptrs inside this obj 
                                         // Poll is an enum: ready (result), pending
        fn poll(self: Pin<&mut Self>, _cx: &mut Context) -> Poll<Self::Output> {
            Poll::Ready(self.n)
        }
    }
    
    // to run the future, use an executor, e.g.: 
    use futures::executer::block_on(); //block thread til future is complete
    block_on(SimpleFuture{n:10});
```

### Future Combinators

Combinators are cheap bc only calling it takes up work, creating combinators are "lazy" operations. Combinators only wrap future in another one, and doesn't do anything until you call it.

Important trait `FutureExt`: 

- `map` can change result type of future
- `then` chains on a closure f for additional computation on top of future
- `into_stream` turns future into many objects
- `join` combines two futures into one

    // Oneshot Channel: allows async via two different threads
    // This si the equivalent of POLL, but multithreaded
    // f is a Future obj
     
    let (ch_s, ch_r) = oneshot::channel();
    block_on(f.map( move |n| ch_s.send(n+5) ));
    let result = block_on(ch_r).unwrap();

### Async Functions

Async block resolves into an enum. The enum implements Future, and tracks the state of its own variables.

`Await` must always be inside Async block to create a new state on the async enum. It adds state to the resulting future. (it doesn't actually await at the time we write it)

```rust
// Don't have to create Future obj manually
// Create a async function instead that auto handles futures

pub async fn simple_future_fn(p:i32) -> i32 { ... }

block_on(async move {
    let v = simple_future_fn(n).await;
  ch_s.send(v);
});

block_on(async move {
    let r = ch_r.await.unwrap();
    ... 
});
```

### Stream

A Stream (Reader) is a stream of results, like an `Iterator`, with a delay between each value.

Streams look like Futures, but returns a Poll `Option<...>` rather than a enum of Ready(Result). Streams impl the Stream trait.

`Ready(Some(thing))`: means that stream has more to ask for
```rust
pub struct ReadStream { reader: A, buf:[u8;100], }
impl<A:AsyncRead + Unpin> Stream for ReadStream<A> {
    type Item = String;

    // Acts like Iterator, to provide a streat of Results
    fn poll_next(self: Pin<&mut Self>, cx &mut Context) -> Poll<Option<Self::Item>> {
            let up = self.get_mut();
      let r = Pin::new(&mut up.reader)
      match r.poll_read(cx, &mut up.buf){
            Poll::Ready(Ok(len)) => Poll::Ready(Some(String::from_utf8_lossy(&up.buf[..len]).to_string()));
            Poll::Ready(Err(_e)) => Poll::Ready(None), // kind of hacky but ok
            Poll::Pending => Poll::Pending,
      }		
    }
}
```
## Databases

Crate: Diesel

Handles databases operations, builder patterns to auto build db functions.

Downloading Diesel: `cargo install diesel_cli --no-default-features --features postgres`

1. Create `.env` file with database url
2. `diesel setup` creates database, with folder `migrations` with up/down sql files in the initial setup migration.
3. `psql -d dbname`  then `\d` to see empty db.
4. `diesel migration generate create_initial_tables`
5. Create SQL commands in newly migration `up` file, then
6. `diesel migration run` runs the migrations (db schema changes)
7. `diesel migration redo` reverts all applied migrations

```rust
// to use schema.rs in a file:
#[macro_use]    // use diesel macros across entire crate 
extern crate diesel;
pub mod schema; // how it works https://diesel.rs/
```

8. `cargo expand` shows us all the functions that are created by diesel macros.

```rust
//Creating a DB connection in lib.rs based on .env setup
pub fn create_connection() -> Result<PgConnection, failure::Error> {
    dotenv::dotenv().ok();
    Ok(PgConnection::establish(&std::env::var("DATABASE_URL")?)?)
}

// models.rs
use crate::schema::*;

// Model declaration: be really careful with attrs ordering!!
#[derive(Queryable, Debug)]
pub struct User {
    pub id: i32,
    pub name: String,
    password: String,
}

impl User {
    pub fn verify_pass(&self, password: &str) -> bool {
        bcrypt::verify(password, &self.password).unwrap_or(false)
    }
}

// Helper structure to allow db to auto create userids
#[derive(Insertable, Queryable)]
#[table_name = "users"]
pub struct NewUser<'a> {
    name: &'a str,
    password: String, // ok to lose the pointer here
}

pub fn new_user<'a>(name: &'a str, password: &str) -> NewUser<'a> {
    NewUser {
        name, 
        password: bcrypt::hash(password, 7).unwrap()
    }
}

// Adding values to table
let added: User = diesel::insert_into(users::table).values(&user).get_result(&conn)?;

// Reading values from table
let res = users::table.limit(10).load::<User>(&conn)?;

// DB inner joins
let vals = table1::table
             .inner_join(table2::table)
             .inner_join(table3::table)
             .filter(table1::id.eq(id))
                         .select((...)) // select
             .load::<NewStruct>(&conn)?;
```
## Web Servers

**Crate**: Rocket - framework for web services to build websites

`rocket::ignite().launch()`:: creates rocket builder, then starts rocket server

```rust
#![feature(proc_macro_hygiene, decl_macro)]
#[macro_use]
extern crate rocket;
use rocket::response::{Responder, NamedFile};

#[get("/")]
fn root() -> Result<impl Responder<'static>, failure::Error> {
    NamedFile::open("site/static/index.html").map_err(|e| e.into())
}

// Enabling static file routes
#[get("/<path..>")]
fn static_file(path: PathBuf) -> Result<impl Responder<'static>, failure::Error> {
    let path = PathBuf::from("site/static").join(path);     // creates site/static/new_path
    NamedFile::open(path).map_err(|e| e.into())
}

#[post("/login", data="<dt>")] //syntax: quote + <>
fn login(dt: Form<LoginData>, db:DPool) -> Result<impl Responder<'static>, DoodleWebErr> {
    let ldform = dt.into_inner();
    let vals = users::table::filter(users::name.eq(ldform.name)).load::<User>(&db.0)?;
    let user = vals.iter().next().ok_or(DoodleWebErr::UserDoesNotExistError)?;
    if ! user.verify_pass(&ldform.pass) {
        return Err(DoodleWebErr::PasswordError)
    }
    Ok("Password passed") // fine bc String impl Responder
}

fn main() {
    rocket::ignite()      //creates rocket builder
            .mount("/", routes![root, static_file])    // mounts route & fn
            .launch();    // starts rocket server, returns error on every case
```

Custom errors & types in Response
```rust
use failure_derive::Fail; 
// allows custom errors to be returned //response is the actual response being returned
// getting response happens after we return it, friendly to be converted to futures. 
use rocket::response::{Responder, Response};
use rocket::Request;
        // status code, information that passed to client end for handling
use rocket::http::{Status, ContentType};
use std::io::Cursor; // read seeker for our response

#[derive(Fail, Debug)]
pub enum CustomWebErr {
    #[fail(display = "IO Error{}", 0)]
    IOErr(std::io::Error),
}

// makes e.into() possible
impl From<std::io::Error> for CustomWebErr {
    fn from(e:std::io::Error) -> Self {
        CustomWebErr::IOErr(e)
    }
}

// implemented responder to be able to respond with custom error
impl <'r> Responder<'r> for CustomWebErr {
    fn respond_to(self, _:&Request) -> rocket::response::Result<'r> {
        let res = Response::build()
                    .status(Status::InternalServerError)
                    .header(ContentType::Plain)
                    .sized_body(Cursor::new(format!("Error doing loading page: {}", self))).finalize(); //to get this we need to impl seekable
        Ok(res)
    }
}
```

Configure Rocket.toml

```rust
    [global.databases]
    doodlebase = {url = "postgres://acct:pw@localhost/dbname"}
````

### Sessions
```rust
//in session.rs
pub struct Session(Arc<Mutex<HashMap<u64,User>>>);

impl Session {
    pub fn new() -> Self {
        Session(Arc::new(Mutex::new(HashMap::new())))
    }

    pub fn get(&self, k:u64) -> Option<User> {
        self.0.lock().unwrap().get(&k).map(|u| u.clone())
    }
}

// in main.rs
// pub main()
let sess = session::Session::new();
    rocket::ignite()
            .mount("/", routes![...])
            .attach(DP::fairing())
            .manage(sess) // attaches session
            .launch();

// Putting Session
let sess_id = ss.put(user.clone());
cookies.add(Cookie::new("login", sess_id.to_string()));

// Getting Session
let login = cookies.get("login").ok_or(DoodleWebErr::NoCookie)?.value();
let user = st.get(login.parse().map_err(|_| "DoodleWebErr::NoCookie")?)
           .ok_or(DoodleWebErr::NoSession)?;
```

### Static templates

*With Maud: compile time templates*

```rust
// pages that return Result<impl Responder>
Ok(
    html! {     // a maud macro
        (DOCTYPE)
        head { meta charset = "utf-8" }
        body { ... }
    }
)
```

## CLI

**Crate**: clap
```rust
use clap::{clap_app, crate_version};

// Define the CLI commands
let clap = clap_app!(cli_name => 
            (about:"About this CLi")
            (version: crate_version!())
            (author: "Your name here")

                        // CLI with input args
            (@subcommand cmd_1 => 
                (@arg field_1:+required "Field 1 Prompt")
                (@arg field_2:-u +takes_value "Field 2 is optional")
            )

                        // CLI with no args
            (@subcommand cmd_2 => )
        ).get_matches(); //reads input from cli, makes everything fit

// Matching commands
if let Some(ref sub) = clap.subcommand_matches("cmd_1") {
                // access value: 
          sub.value_of("field_1").unwrap()
}
```
