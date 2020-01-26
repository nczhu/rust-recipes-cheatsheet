## TOC 
- [Important Traits](#Important-Traits)
- [Important Crates](#Important-Crates)
- [Important Shorthands](#Important-Shorthands)
- [Stack vs Heap](#Stack-vs-Heap)
- [Strings](#Strings)
- [Vectors](#Vectors)
- [Return Handling](#Return-Handling)
- [Iterators](#Iterators)
- [Generics](#Generics)
- [Lifetimes](#Lifetimes)
- [Closures](#Closures)
- [References](#References)
- [Threads](#Threads)
- [Channels](#Channels)
- [Patterns](#Patterns)
- [Macros](#Macros)
- [Futures](#Futures)
- [Databases](#Databases)
- [Web Services](#Web-Services)

## Important Traits

- `from` and `into`: *Two sides of the same coin, returns the same Type the trait is impl on. `Into` is free after impl `From`*

    let my_string = String::from(my_str);
    let num: Number = int.into();

- `PartialOrd`: Allows the > , < , = operators in T
- `std::ops::AddAssign`: Allows arithmetic operations in T
- `Copy`: Allows copy for primitives, rather than just move
- `Default`: `.. Default::default()` fills in defaults for a struct/obj;

## Important Crates

- `Failure`: makes custom error creation easier
- `Itertools`: interleave(), interlace() *for strings* , combines iterators together in woven way
- `Rayon`: Thread, pool handling by allowing **parallel iterators.** `intoParallelIterator`

## Important Shorthands

- `..`: inclusive range operator,  e.g. `start..=end`
- `..`: struct update syntax, i.e. remaining fields not explicitly set should have the same value as the fields in the given instance. `..Default::default()` or `..user1`

## Stack vs Heap

- `Stack`: Just the vars that exist only during a function
- `Heap`: Memory that's generally available. Need to free it before ending program. e.g. `String` , `Box`, `Hashmap`, `Arc`, `Rc`
    - `Box`: A pointer type for heap allocation. Allocates memory on the heap and then places x/anything into it. When it's **dropped**, the memory it points to will be freed.

## Strings

### Str Types

- `String`: Dynamic string heap type. *Acts as a pointer. Use when I need owned string data (passing to other threads, building at runtime)*
- `str`: immutable seq of bytes of dynamic/unknown length in memory. Commonly ref by `&str`, e.g. a ptr to a slice of data that can be anywhere. *Use when I only need a view of a string.*
- `&'static str`: e.g. "foo" string literal, hardcoded into executable, loaded into mem when program runs.

### Usage Conventions

- Trait `AsRef<str>` : if S (generic var) is already a string, then it returns pointer to itself, but if it's already a pointer to str, then it doesn't do anything (no op). Has method `.as_ref()` *Saves me from implementing fns for both strings and slices*
- `format!("a {}", n)`: returns string, concatenates
- **Ptr to String → Option<String>**: `ptr.map(|s| s.to_string())`
- **String slice into a Result<expected type>**: `.parse()`

## Vectors

    // Common uses
    let mut v = Vec::with_capacity(100);
    v.split_at_mut(index); // splits vec returning mut vecs

## Return Handling

### Result: Ok vs Error

    pub enum Result<T,E> {
    	Ok(T),
    	Err(E),
    }

### ? Shorthand

*? applies to a `Result` value, and if it was an `Ok`, it **unwraps** and gives the inner value (Directly uses the into() method), or it returns the Err type from the current* function.* 

    // Unwraps a result, or propagates an error (returns the `into` of an Error)
    // ? needs the From trait implementation for the error type you are converting
    Ok(serde_json::from_str(&std::fs::read_to_string(fname)?)?)

### Use custom errors for new structs

*To handle proper error handling*

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

### Option: Some vs None

    pub enum Option<T> {
    	Some(T),
    	None,
    }

### Result <> Option

    // Result into Option
    let t = get_result.ok()?;
    
    // Option into Result
    // ok_or converts None into Error
    // ? gets it into the Err type we want for our function
    let t = get_option.ok_or("An Error message if the Option is None")?;

### Refactor error handling

1. Put all error traits & implementations in `error.rs`; Put all functions in `lib.rs`
2. Create a lib in cargo.toml, point [main.rs](http://main.rs) to library crate  `extern crate lib_name`

    // Allows us to have both a library and an application
    [lib]     # library name, path, etc
    [[bin]]   # binary target name, path etc. Can have multiple binaries!

3. Inside `[lib.rs](http://lib.rs)` import error.rs with `mod error`  `use error::...`

**Failure Trait:** *Better failure handling, without having to create my own error type*

[failure](https://rust-lang-nursery.github.io/failure/)

- When using it to write a library: create a new Error type
- When using it to write an application: have all functions that can fail return failure::Error

## Iterators

Important iterating traits: 

- `Iterator`: Requires Item type, `.next()`
- `IntoIterator`: Key trait that allows for loop to work, for things that doesn't naturally iterate over themselves
- `a_vec.into_iter()`: turns a vector into an iterator

## Generics

*Generic Type, Functions & Trait imps*

    // TRAITS
    pub struct Thing<T> { foo: T, ...}...
    impl <T> Thing<T> {...}
    impl <T> TraitA for Thing<T> {...}
    
    // Limit generics to certain Traits & provide addn traits it needs
    impl <T:TraitA + TraitB> ...
    // Or
    impl <T> ... where T:SomeTrait {...}
    
    // 1. Refactor the multi-traits by creating a new Trait
    pub trait CombiTrait:TraitA + TraitB {..// any fns ..}
    // 2. Impl CombiTrait for all T that has traitA/B, otherwise get error
    impl <T:TraitA + TraitB> CombiTrait for T {}

    // FUNCTIONS
    // It makes the type available for the function
    fn print_c<I:TraitA>(foo:I) {

## Lifetimes

*Think of lifetimes as a constraint, bounding output to the same lifetime as its input(s)*

- By default, lifetimes exist only within the scope of a function
- Only things relating to references need a lifetime, e.g. struct containing a ref, or strings
- `'static` ensures a lifetime for the entirety of the program

    ```
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

## Closures

- `FnOnce`: consumes variables it captures. Moves vars into the closure when its defined. FnOnce can only be called once on the same variables
- `FnMut`: can change env bc it mutably borrows values. *recommended for working with Vecs*
- `Fn`: borrow values from env immutably

    // FnOnce:
    pub fn foo<F>(&mut self, f: F)
    	where F: FnOnce(&mut String), {...}
    
    // FnMut: The function itself is going to change over time
    pub fn edit_all<F>(&mut self, mut f: F)
    	where F: FnMut(&mut String), {...}

    // Boxed closure type, usually used in threads, 
    // allowing different fn to be sent to the same channel
    Box<Fn(&mut String) + TraitB >

## References

    &i32        // a reference
    &'a i32     // a reference with an explicit lifetime
    &'a mut i32 // a mutable reference with an explicit lifetime
    
    Box<T>      // is for single ownership.
    Rc<T>       // is for multiple ownership.
    Arc<T>      // is for multiple ownership, but threadsafe.
    Cell<T>     // is for “interior mutability” for Copy types; that is, when you need to mutate something behind a &T.

### Reference count

*Single thread use only: doesn't impl sync and send*

- `Rc<>`: generic struct, like a garbage collector.  `(reference count, ptr to data)` When you clone an Rc, it just increments the count without copying the data; when it reaches 0 it drops its internal pointer.
- `Rc::new(s)`: Turns into reference counted obj, & acts as a reference, usable with `.clone()`. S has to be immutable.

    // Make sure to specific the Rc type when a var is going to be reference counted
    s: Rc<String> // a generic type, acts as a reference to String

### RefCell: a mutable borrow

*Single thread use only: doesn't impl sync and send*

- Internally: guards the borrow checker, makes sure changes are correct.
- Externally: pretends that it doesn't change. To the borrow checker, it mimics immutable obj.
- `Rc<<RefCell<s>>` , `Rc::new(RefCell::new(s))`, `let ... = s.borrow_mut()` and lastly `drop(s)` to disable the guard.

## Threads

- Main fn can terminate before new threads finish!

    use std::thread::*;
    use std::time::Duration;
    
    // Spinning up new thread
    fn foo() {
    	// closure for new thread actions
    	spawn( || { println!("This is the new channel"); });
    	println!("This is the initial thread");
    	// allows time for new thread to finish before quitting fn
    	sleep(Duration::from_millis(1000));
    }

    // Diff way to wait for threads to complete
    let mut children = Vec::new();
    let child = thread::spawn(...);
    for child in children { child.join().expect("thread panic");

- When you need to move **primitive** variables to be accessible inside threads, thanks to `Copy`:

    spawn( move || { println!("{}", n); });

### Arc, Mutex

If `Copy` is not implemented for the var (e.g. moving a String btw threads), use: 

- `arc`: atomic reference count.
- `mutex`: a guard on mutation, allows mutation when ppl have a `lock` on it
- `lock()`: acquires the mutex's lock, ensuring its only held by 1 obj, returns Result<>.

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

When to use Mutex in Arc?

    // In a multithread game engine, you might want to make an Arc<Player>. 
    // The Mutexes protect the inner fields that are able to change (items). 
    // while still allowing multiple threads to access id concurrently!
    
    struct Player {
        id: String,
        items: Mutex<Items>,
    }

### Channels

- A function returning: `Sender<T>` and `Receiver<T>` endpoints, where T is type of msg to be transferred, allowing async communications between threads
- `mpsc`: means Multi Producer Single Consumer. We can clone sender, but not receiver.
- Drop(channel senders) will close the channel automatically
- **Fns have to be Boxed** before sending across channel

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

### ThreadPools

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

## Patterns

### Builder

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

### Defaults

    struct Foo {...}
    
    impl Default for Foo {
    	fn default() -> Self {
    		Foo { ..:.., }
    	}
    }
    ...
    let f = Foo { ..Default:default()}

### Wrappers Traits

If I want to compose structs that contain certain (different?) traits. Create a wrapper struct. *This is usually done with iterators etc.*

    pub struct TraitWrap<A:TraitA, B:TraitB> {
        a: A,
        b: B,
    }
    
    impl <A: TraitA, B:TraitB> TraitA for TraitWrap<A,B> {
    		fn ... 
    }

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

Macros that work like match statements on your code snippets to expand them into finished code.

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
    

### Macro refactoring

Calling a macro within a macro is a good way to refactor the macro code

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

### Procedural Macros

Functions that operate on input code, outputs new code

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

### Derive Macros

**Good convention**: for structs, to create setter helper fn, e.g. `set_field1` `set_field2` .

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

- `cargo expand`: expands macro code to be more readable. Install through `cargo install cargo-expand`

## Futures

## Databases

## Web Services